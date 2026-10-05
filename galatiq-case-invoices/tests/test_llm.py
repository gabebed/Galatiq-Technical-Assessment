"""Offline tests for the LLM module (no network; the SDK client is faked)."""

import json

import grpc
import pytest
from pydantic import BaseModel, ValidationError

from invoice_processor.llm import LLMError, LLMSettings, XAIClient, create_llm_client

FAKE_KEY = "xai-test-not-a-real-key"


class Answer(BaseModel):
    result: int


class FakeResponse:
    class usage:  # noqa: N801 - mimics SDK attribute
        prompt_tokens = 10
        completion_tokens = 2

    cost_usd = 0.0


class FakeChat:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.messages: list = []

    def append(self, message) -> None:
        self.messages.append(message)

    def parse(self, schema):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return FakeResponse(), schema.model_validate(self.outcome)


class FakeSDK:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.created: list[dict] = []
        self.chat = self

    def create(self, **kwargs) -> FakeChat:
        self.created.append(kwargs)
        self.last_chat = FakeChat(self.outcome)
        return self.last_chat


class FakeRpcError(grpc.RpcError):
    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.UNAUTHENTICATED

    def details(self) -> str:
        return "invalid API key"


# --------------------------------------------------------------------------- #
# Settings and factory
# --------------------------------------------------------------------------- #


def test_settings_defaults() -> None:
    assert LLMSettings.from_env({}) == LLMSettings(provider="xai", model="grok-4.7", timeout_seconds=60.0)


def test_settings_from_env() -> None:
    settings = LLMSettings.from_env({"LLM_PROVIDER": " XAI ", "LLM_MODEL": "grok-4.3", "LLM_TIMEOUT_SECONDS": "15"})
    assert settings == LLMSettings(provider="xai", model="grok-4.3", timeout_seconds=15.0)


@pytest.mark.parametrize("timeout", ["abc", "0", "-1"])
def test_invalid_timeout(timeout: str) -> None:
    with pytest.raises(LLMError, match="LLM_TIMEOUT_SECONDS"):
        LLMSettings.from_env({"LLM_TIMEOUT_SECONDS": timeout})


def test_unknown_provider() -> None:
    with pytest.raises(LLMError, match="Unknown LLM provider 'acme'"):
        create_llm_client(env={"LLM_PROVIDER": "acme"})


@pytest.mark.parametrize("env", [{}, {"XAI_API_KEY": "   "}])
def test_missing_api_key(env: dict) -> None:
    with pytest.raises(LLMError, match="XAI_API_KEY is not set"):
        create_llm_client(env=env)


def test_factory_builds_xai_client_from_env() -> None:
    client = create_llm_client(env={"XAI_API_KEY": FAKE_KEY, "LLM_MODEL": "grok-4.3"})
    assert isinstance(client, XAIClient)
    assert client.settings.model == "grok-4.3"
    assert FAKE_KEY not in repr(client)


# --------------------------------------------------------------------------- #
# XAIClient behaviour
# --------------------------------------------------------------------------- #


def test_complete_structured_returns_parsed_model() -> None:
    sdk = FakeSDK({"result": 5})
    client = XAIClient(LLMSettings(model="grok-4.7"), sdk_client=sdk)

    assert client.complete_structured("Be precise.", "What is 2 + 3?", Answer) == Answer(result=5)
    assert sdk.created == [{"model": "grok-4.7"}]
    roles = [m.role for m in sdk.last_chat.messages]
    assert len(roles) == 2 and roles[0] != roles[1]  # system then user
    assert "Be precise." in str(sdk.last_chat.messages[0])
    assert "What is 2 + 3?" in str(sdk.last_chat.messages[1])


def test_rpc_errors_become_llm_errors() -> None:
    client = XAIClient(LLMSettings(), sdk_client=FakeSDK(FakeRpcError()))
    with pytest.raises(LLMError, match=r"UNAUTHENTICATED.*invalid API key"):
        client.complete_structured("s", "p", Answer)


# --------------------------------------------------------------------------- #
# Tool-calling loop
# --------------------------------------------------------------------------- #


class _Function:
    def __init__(self, name: str, arguments: str) -> None:
        self.name, self.arguments = name, arguments


class _ToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id, self.function = call_id, _Function(name, arguments)


class _Sampled:
    def __init__(self, tool_calls: list) -> None:
        self.tool_calls = tool_calls


class ToolChat(FakeChat):
    def __init__(self, rounds: list[list[_ToolCall]], final: dict) -> None:
        super().__init__(final)
        self.rounds = list(rounds)

    def sample(self):
        return _Sampled(self.rounds.pop(0) if self.rounds else [])


class ToolSDK(FakeSDK):
    def __init__(self, rounds, final) -> None:
        super().__init__(final)
        self.rounds = rounds

    def create(self, **kwargs) -> ToolChat:
        self.created.append(kwargs)
        self.last_chat = ToolChat(self.rounds, self.outcome)
        return self.last_chat


def _echo_toolset():
    from invoice_processor.tools import Tool, ToolArgs, Toolset

    class Args(ToolArgs):
        text: str

    return Toolset("test", [Tool("echo", "Echo text.", Args, lambda a: Answer(result=len(a.text)))])


def test_run_tools_routes_every_call_through_the_toolset() -> None:
    rounds = [[_ToolCall("c1", "echo", '{"text": "abc"}'), _ToolCall("c2", "exec", '{"code": "1"}')]]
    sdk = ToolSDK(rounds, {"result": 7})
    toolset = _echo_toolset()

    answer = XAIClient(LLMSettings(), sdk_client=sdk).run_tools("s", "p", toolset, Answer)

    assert answer == Answer(result=7)
    assert [(c.tool, c.ok) for c in toolset.calls] == [("echo", True), ("exec", False)]
    assert len(sdk.created[0]["tools"]) == 1 and sdk.created[0]["tools"][0].function.name == "echo"
    tool_messages = [m for m in sdk.last_chat.messages if getattr(m, "tool_call_id", "")]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
    refused = json.loads(tool_messages[1].content[0].text)
    assert refused["ok"] is False and "not available" in refused["error"]


def test_run_tools_stops_runaway_loops() -> None:
    rounds = [[_ToolCall(f"c{i}", "echo", '{"text": "x"}')] for i in range(10)]
    client = XAIClient(LLMSettings(), sdk_client=ToolSDK(rounds, {"result": 1}))
    with pytest.raises(LLMError, match="exceeded 3 tool-calling rounds"):
        client.run_tools("s", "p", _echo_toolset(), Answer, max_rounds=3)


def test_schema_mismatch_becomes_llm_error() -> None:
    client = XAIClient(LLMSettings(), sdk_client=FakeSDK({"result": "not a number"}))
    with pytest.raises(LLMError, match="did not match Answer"):
        client.complete_structured("s", "p", Answer)
    # Sanity: the fake really raises a ValidationError underneath.
    with pytest.raises(ValidationError):
        Answer.model_validate({"result": "not a number"})
