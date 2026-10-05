"""Per-request LLM telemetry.

Provider clients wrap every network request in ``observe_request``, which logs
immediately before and after the call (logger ``invoice_processor.llm.calls``)
and produces an ``LLMCallRecord``. Records are also appended to the active
``collect_llm_calls()`` collector so callers (the CLI) can show exactly which
requests were made for each invoice.

Never recorded: API keys, prompts, or model output. Errors are reduced to a type,
status code, and (for schema failures) the failing field paths, never the values.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger("invoice_processor.llm.calls")

R = TypeVar("R")
StructuredStatus = Literal["parsed", "invalid", "not_requested", "not_received"]

_sequence = itertools.count(1)
_collector: ContextVar[list[LLMCallRecord] | None] = ContextVar("llm_call_collector", default=None)
_MAX_ERROR_CHARS = 200


class LLMCallRecord(BaseModel):
    """One request to an LLM provider, as observed by the client wrapper."""

    model_config = ConfigDict(frozen=True)

    sequence: int
    agent: str
    provider: str
    model: str
    operation: str  # "structured" | "tool_round" | "final_structured"
    schema_name: str | None = None
    started_at: datetime
    completed_at: datetime
    duration_ms: float
    api_ok: bool
    structured_output: StructuredStatus
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    tool_calls: int | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.api_ok and self.structured_output in ("parsed", "not_requested")


@contextmanager
def collect_llm_calls() -> Iterator[list[LLMCallRecord]]:
    """Collect every LLM request made inside the block (e.g. while processing one invoice)."""
    calls: list[LLMCallRecord] = []
    token = _collector.set(calls)
    try:
        yield calls
    finally:
        _collector.reset(token)


def describe_error(exc: BaseException) -> str:
    """A short, content-free description of a request failure."""
    if isinstance(exc, ValidationError):
        locations = sorted({".".join(map(str, e["loc"])) or "<root>" for e in exc.errors()})
        return f"schema validation failed ({exc.error_count()} error(s) at: {', '.join(locations)})"
    if isinstance(exc, json.JSONDecodeError):
        return f"response was not valid JSON (position {exc.pos})"
    code = getattr(exc, "code", None)
    if callable(code):  # grpc.RpcError
        details = exc.details() if callable(getattr(exc, "details", None)) else ""
        return f"{code().name}: {details}"[:_MAX_ERROR_CHARS]
    return f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]


def observe_request(
    call: Callable[[], R],
    *,
    agent: str,
    provider: str,
    model: str,
    operation: str,
    schema_name: str | None,
    is_schema_error: Callable[[BaseException], bool],
    describe_response: Callable[[R], dict[str, Any]],
    structured: bool,
) -> R:
    """Run one provider request, logging before and after it and recording the outcome.

    Exceptions are re-raised unchanged after being recorded.
    """
    sequence = next(_sequence)
    base = {"sequence": sequence, "agent": agent, "provider": provider, "model": model,
            "operation": operation, "schema_name": schema_name}
    logger.info("LLM request #%d START  agent=%s provider=%s model=%s op=%s%s", sequence, agent, provider, model,
                operation, f" schema={schema_name}" if schema_name else "", extra={"llm": base | {"event": "start"}})

    started_at, started = datetime.now(timezone.utc), time.perf_counter()
    outcome: dict[str, Any]
    try:
        result = call()
    except BaseException as exc:
        schema_failure = is_schema_error(exc)
        outcome = {"api_ok": schema_failure,
                   "structured_output": "invalid" if schema_failure else "not_received",
                   "error": describe_error(exc)}
        _finish(base, started_at, started, outcome)
        raise
    outcome = {"api_ok": True, "structured_output": "parsed" if structured else "not_requested",
               **describe_response(result)}
    _finish(base, started_at, started, outcome)
    return result


def _finish(base: dict[str, Any], started_at: datetime, started: float, outcome: dict[str, Any]) -> None:
    record = LLMCallRecord(
        **base, **outcome, started_at=started_at, completed_at=datetime.now(timezone.utc),
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    if (calls := _collector.get()) is not None:
        calls.append(record)

    tokens = (f" tokens={record.prompt_tokens}->{record.completion_tokens}"
              if record.prompt_tokens is not None else "")
    cost = f" cost_usd={record.cost_usd:.6f}" if record.cost_usd is not None else ""
    request_id = f" id={record.request_id}" if record.request_id else ""
    tools = f" tool_calls={record.tool_calls}" if record.tool_calls else ""
    status = "OK" if record.success else "FAILED"
    message = (f"LLM request #{record.sequence} {status} agent={record.agent} provider={record.provider} "
               f"model={record.model} op={record.operation} structured={record.structured_output}"
               f"{request_id}{tokens}{cost}{tools} duration_ms={record.duration_ms:.0f}")
    if record.error:
        message += f" error={record.error}"
    payload = {"llm": record.model_dump(mode="json") | {"event": "end", "success": record.success}}
    logger.log(logging.INFO if record.success else logging.ERROR, message, extra=payload)
