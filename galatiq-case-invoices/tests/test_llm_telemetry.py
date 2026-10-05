"""LLM request telemetry, observed through the real XAIClient with only the network transport faked."""

import json
import logging
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest
from pydantic import BaseModel

from invoice_processor import cli as cli_module
from invoice_processor.cli import main
from invoice_processor.database import init_db
from invoice_processor.llm import LLMError, LLMSettings, XAIClient
from invoice_processor.llm.telemetry import collect_llm_calls
from invoice_processor.llm.telemetry import logger as telemetry_logger
from invoice_processor.observability import JsonFormatter
from invoice_processor.tools import Tool, ToolArgs, Toolset

INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"
FAKE_KEY = "xai-TEST-KEY-must-never-appear-in-logs"
SECRET_PROMPT = "CONFIDENTIAL invoice note: account 12345678"


class Answer(BaseModel):
    result: int


class Unavailable(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "connection refused"


class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class FakeResponse:
    def __init__(self, tool_calls=(), request_id="resp-7f3a"):
        self.id = request_id
        self.usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=80, reasoning_tokens=30, total_tokens=1310)
        self.cost_usd = 0.0042
        self.tool_calls = list(tool_calls)


class FakeChat:
    def __init__(self, sdk, tools):
        self.sdk, self.tools = sdk, tools
        self.pending = list(sdk.tool_calls.get(tuple(sorted(tools)), []))

    def append(self, message):
        pass

    def sample(self):
        self.sdk.events.append("sdk:sample")
        if self.sdk.error:
            raise self.sdk.error
        calls, self.pending = self.pending, []
        return FakeResponse(tool_calls=calls)

    def parse(self, schema):
        self.sdk.events.append(f"sdk:parse:{schema.__name__}")
        if self.sdk.error:
            raise self.sdk.error
        return FakeResponse(), schema.model_validate(self.sdk.answers[schema.__name__])


class FakeSDK:
    """Stands in for xai_sdk.Client: records each network call in ``events``."""

    def __init__(self, answers=None, error=None, tool_calls=None):
        self.answers, self.error = answers or {}, error
        self.tool_calls = tool_calls or {}
        self.events: list[str] = []
        self.chat = self

    def create(self, **kwargs):
        return FakeChat(self, [t.function.name for t in kwargs.get("tools") or []])


class EventHandler(logging.Handler):
    """Puts telemetry log lines into the same event list as the SDK calls, to check ordering."""

    def __init__(self, events, records):
        super().__init__()
        self.events, self.records = events, records

    def emit(self, record):
        self.records.append(record)
        self.events.append("log:START" if " START " in record.getMessage() else "log:END")


@pytest.fixture
def capture():
    """Attach a handler to the telemetry logger; returns the captured records for a FakeSDK."""
    saved_level, handlers = telemetry_logger.level, []
    telemetry_logger.setLevel(logging.INFO)  # setLevel (not attribute assignment) also clears the level cache

    def attach(sdk) -> list[logging.LogRecord]:
        records: list[logging.LogRecord] = []
        handler = EventHandler(sdk.events, records)
        telemetry_logger.addHandler(handler)
        handlers.append(handler)
        return records

    yield attach
    for handler in handlers:
        telemetry_logger.removeHandler(handler)
    telemetry_logger.setLevel(saved_level)


def _client(sdk) -> XAIClient:
    client = XAIClient(LLMSettings(model="grok-4.7"), env={"XAI_API_KEY": FAKE_KEY})
    client._client = sdk  # real wrapper, fake transport
    return client


# =========================================================================== #
# One record and two log lines per real request, emitted around the API call
# =========================================================================== #


def test_successful_request_is_logged_before_and_after_the_api_call(capture) -> None:
    sdk = FakeSDK(answers={"Answer": {"result": 5}})
    records = capture(sdk)

    with collect_llm_calls() as calls:
        assert _client(sdk).complete_structured("Be precise.", SECRET_PROMPT, Answer, agent="test-agent") == Answer(result=5)

    assert sdk.events == ["log:START", "sdk:parse:Answer", "log:END"]
    [call] = calls
    assert (call.agent, call.provider, call.model, call.operation, call.schema_name) == (
        "test-agent", "xai", "grok-4.7", "structured", "Answer")
    assert call.api_ok and call.structured_output == "parsed" and call.success
    assert call.request_id == "resp-7f3a"
    assert (call.prompt_tokens, call.completion_tokens, call.reasoning_tokens, call.total_tokens) == (1200, 80, 30, 1310)
    assert call.cost_usd == pytest.approx(0.0042)
    assert call.started_at <= call.completed_at and call.duration_ms >= 0
    end = records[-1].getMessage()
    for fragment in ("OK", "agent=test-agent", "provider=xai", "model=grok-4.7", "id=resp-7f3a",
                     "tokens=1200->80", "structured=parsed"):
        assert fragment in end


def test_invalid_structured_output_is_recorded_without_its_content(capture) -> None:
    sdk = FakeSDK(answers={"Answer": {"result": "SENSITIVE-VALUE-FROM-MODEL"}})
    records = capture(sdk)

    with collect_llm_calls() as calls, pytest.raises(LLMError) as raised:
        _client(sdk).complete_structured("s", "p", Answer, agent="test-agent")

    [call] = calls
    assert call.api_ok is True and call.structured_output == "invalid" and not call.success
    assert "result" in call.error  # the failing field path...
    assert "SENSITIVE-VALUE-FROM-MODEL" not in call.error  # ...but never the value
    assert "SENSITIVE-VALUE-FROM-MODEL" not in str(raised.value)
    assert records[-1].levelno == logging.ERROR and "FAILED" in records[-1].getMessage()


def test_api_failure_is_recorded_and_raised(capture) -> None:
    sdk = FakeSDK(error=Unavailable())
    records = capture(sdk)

    with collect_llm_calls() as calls, pytest.raises(LLMError, match="UNAVAILABLE: connection refused"):
        _client(sdk).complete_structured("s", "p", Answer, agent="test-agent")

    [call] = calls
    assert call.api_ok is False and call.structured_output == "not_received"
    assert call.error == "UNAVAILABLE: connection refused"
    assert call.request_id is None and call.prompt_tokens is None
    assert sdk.events == ["log:START", "sdk:parse:Answer", "log:END"]
    assert records[-1].levelno == logging.ERROR


def test_every_tool_round_is_a_separate_recorded_request(capture) -> None:
    class Args(ToolArgs):
        text: str

    toolset = Toolset("lookup", [Tool("echo", "Echo.", Args, lambda a: Answer(result=len(a.text)))])
    call = SimpleNamespace(id="c1", function=_Fn("echo", '{"text": "abc"}'))
    sdk = FakeSDK(answers={"Answer": {"result": 3}}, tool_calls={("echo",): [call]})
    capture(sdk)

    with collect_llm_calls() as calls:
        _client(sdk).run_tools("s", "p", toolset, Answer)

    assert [(c.agent, c.operation, c.tool_calls) for c in calls] == [
        ("lookup", "tool_round", 1), ("lookup", "tool_round", 0), ("lookup", "final_structured", 0)]
    assert [c.structured_output for c in calls] == ["not_requested", "not_requested", "parsed"]
    assert sdk.events == ["log:START", "sdk:sample", "log:END"] * 2 + ["log:START", "sdk:parse:Answer", "log:END"]


def test_no_api_key_or_prompt_content_is_logged(capture) -> None:
    sdk = FakeSDK(answers={"Answer": {"result": 5}})
    records = capture(sdk)
    with collect_llm_calls() as calls:
        _client(sdk).complete_structured(SECRET_PROMPT, SECRET_PROMPT, Answer, agent="test-agent")

    rendered = [r.getMessage() for r in records] + [JsonFormatter().format(r) for r in records]
    rendered += [c.model_dump_json() for c in calls]
    for text in rendered:
        assert FAKE_KEY not in text
        assert "CONFIDENTIAL" not in text and "12345678" not in text


def test_json_logs_carry_structured_llm_fields(capture) -> None:
    sdk = FakeSDK(answers={"Answer": {"result": 5}})
    records = capture(sdk)
    _client(sdk).complete_structured("s", "p", Answer, agent="approval:draft")

    start, end = (json.loads(JsonFormatter().format(r))["llm"] for r in records)
    assert start["event"] == "start" and start["agent"] == "approval:draft" and start["model"] == "grok-4.7"
    assert end["event"] == "end" and end["success"] is True and end["request_id"] == "resp-7f3a"
    assert end["structured_output"] == "parsed" and end["prompt_tokens"] == 1200


# =========================================================================== #
# CLI: it is obvious when Grok is called, and API failures are never silent
# =========================================================================== #

GOOD_ANSWERS = {
    "ValidationReview": {"summary": "All items are in stock."},
    "ApprovalDraft": {"decision": "approved", "requires_additional_scrutiny": False,
                      "reasoning": "Total $5,000.00 is not greater than the $10,000.00 threshold; validation "
                                   "passed with no errors or warnings. Approve."},
    "Critique": {"findings": []},
    "PaymentReport": {"summary": "Paid."},
}
PAY_1001 = SimpleNamespace(id="p1", function=_Fn("mock_payment", '{"vendor": "Widgets Inc.", "amount": "5000.00"}'))


def _run_cli(monkeypatch, capsys, tmp_path, sdk, *extra: str) -> tuple[int, str, str]:
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(cli_module, "create_llm_client", lambda: _client(sdk))
    db = init_db(tmp_path / "inventory.db")
    code = main([f"--invoice_path={INVOICES / 'invoice_1001.txt'}", f"--db-path={db}", "--llm=on", *extra])
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    saved = root.handlers[:], root.level, telemetry_logger.level
    yield
    root.handlers[:], root.level = saved[0], saved[1]
    telemetry_logger.setLevel(saved[2])


def test_cli_shows_every_real_grok_request(monkeypatch, capsys, tmp_path, restore_logging) -> None:
    sdk = FakeSDK(answers=GOOD_ANSWERS, tool_calls={("mock_payment",): [PAY_1001]})
    code, out, err = _run_cli(monkeypatch, capsys, tmp_path, sdk)

    assert code == 0
    network_calls = [e for e in sdk.events if e.startswith("sdk:")]
    starts = [line for line in err.splitlines() if " START " in line]
    ends = [line for line in err.splitlines() if " OK " in line]
    assert len(starts) == len(ends) == len(network_calls)  # one START/OK pair per real request
    assert all("provider=xai model=grok-4.7" in line for line in starts)
    assert {line.split("agent=")[1].split()[0] for line in starts} == {
        "validation", "approval:draft", "approval:critic", "payment"}

    assert f"LLM CALLS  {len(network_calls)} request(s) to xai/grok-4.7: {len(network_calls)} ok, 0 failed" in out
    assert "request id resp-7f3a" in out and " parsed " in out
    assert FAKE_KEY not in out + err
    assert "warning:" not in err


def test_cli_makes_api_failures_visible(monkeypatch, capsys, tmp_path, restore_logging) -> None:
    sdk = FakeSDK(error=Unavailable())
    code, out, err = _run_cli(monkeypatch, capsys, tmp_path, sdk)

    assert code == 3  # the payment agent could not run, so the invoice is not paid
    assert "LLM FAILURES" in out
    assert "Validation agent failed; the deterministic validation result was used" in out
    assert "Approval agent failed; the deterministic policy decision was used" in out
    assert "Payment agent failed; the invoice was NOT paid" in out
    assert "error: UNAVAILABLE: connection refused" in out
    assert "warning: 3 of 3 LLM request(s) failed; fallback or failure in 1 invoice(s): invoice_1001.txt" in err
    assert sum("FAILED" in line and "UNAVAILABLE" in line for line in err.splitlines()) == 3


def test_cli_warns_when_llm_cannot_be_started_in_auto_mode(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_PROVIDER", "acme")
    db = init_db(tmp_path / "inventory.db")
    code = main([f"--invoice_path={INVOICES / 'invoice_1001.txt'}", f"--db-path={db}"])
    err = capsys.readouterr().err

    assert code == 0
    assert "warning: LLM agents disabled, running deterministically: Unknown LLM provider 'acme'" in err


def test_deterministic_mode_makes_no_llm_requests(capsys, tmp_path) -> None:
    db = init_db(tmp_path / "inventory.db")
    with collect_llm_calls() as calls:
        main([f"--invoice_path={INVOICES / 'invoice_1001.txt'}", f"--db-path={db}", "--llm=off"])
    out = capsys.readouterr().out
    assert calls == [] and "LLM CALLS" not in out and "mode: deterministic (no LLM)" in out


# =========================================================================== #
# --llm-log / --no-llm-log
# =========================================================================== #


def test_no_llm_log_hides_successful_requests(monkeypatch, capsys, tmp_path, restore_logging) -> None:
    sdk = FakeSDK(answers=GOOD_ANSWERS, tool_calls={("mock_payment",): [PAY_1001]})
    code, out, err = _run_cli(monkeypatch, capsys, tmp_path, sdk, "--no-llm-log")

    assert code == 0 and "PAYMENT  PAID" in out  # Grok was still used...
    assert any(e.startswith("sdk:") for e in sdk.events)
    assert "LLM request" not in err and "LLM CALLS" not in out  # ...but not logged per request
    assert "request logging off, failures only" in out


def test_no_llm_log_keeps_llm_calls_in_json_output(monkeypatch, capsys, tmp_path, restore_logging) -> None:
    sdk = FakeSDK(answers=GOOD_ANSWERS, tool_calls={("mock_payment",): [PAY_1001]})
    code, out, _ = _run_cli(monkeypatch, capsys, tmp_path, sdk, "--no-llm-log", "--json")
    [invoice] = json.loads(out)["invoices"]
    assert code == 0 and len(invoice["llm_calls"]) == len([e for e in sdk.events if e.startswith("sdk:")])


def test_no_llm_log_still_shows_failures(monkeypatch, capsys, tmp_path, restore_logging) -> None:
    code, out, err = _run_cli(monkeypatch, capsys, tmp_path, FakeSDK(error=Unavailable()), "--no-llm-log")

    assert code == 3
    assert " START " not in err  # routine lines are hidden...
    assert sum("FAILED" in line and "UNAVAILABLE" in line for line in err.splitlines()) == 3  # ...failures are not
    assert "LLM FAILURES" in out and "Payment agent failed; the invoice was NOT paid" in out
    assert "LLM CALLS" not in out
    assert "warning: 3 of 3 LLM request(s) failed" in err


def test_llm_log_is_on_by_default(monkeypatch, capsys, tmp_path, restore_logging) -> None:
    sdk = FakeSDK(answers=GOOD_ANSWERS, tool_calls={("mock_payment",): [PAY_1001]})
    _, out, err = _run_cli(monkeypatch, capsys, tmp_path, sdk)
    assert "LLM request #" in err and "LLM CALLS" in out and "request logging on" in out
