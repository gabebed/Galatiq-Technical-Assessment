"""Tool-calling framework for LLM agents.

An agent receives a ``Toolset``, never raw functions. ``Toolset.invoke`` is the
only way a tool runs, and it:

* allows only the tools registered in that toolset (per-agent allowlist),
* validates arguments against a strict Pydantic model (unknown fields rejected),
* enforces a per-toolset call budget,
* logs every call and keeps an audit record (``Toolset.calls``),
* returns a structured ``ToolResult`` instead of raising, so the agent sees
  refusals and errors as data.

There is no generic code-execution or SQL tool: agents can only call the
specific, typed operations they are given.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

A = TypeVar("A", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)

_MAX_LOGGED_ARGUMENTS = 1000


class ToolArgs(BaseModel):
    """Base for tool argument models: immutable, no unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ToolRefused(Exception):
    """Raised by a tool handler to refuse a call; the message is returned to the agent."""


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: str
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class ToolCallRecord(BaseModel):
    """Audit record of one tool invocation."""

    model_config = ConfigDict(frozen=True)

    toolset: str
    tool: str
    arguments: str
    ok: bool
    error: str | None = None
    duration_ms: float
    called_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class Tool(Generic[A, R]):
    name: str
    description: str
    args_model: type[A]
    handler: Callable[[A], R]

    def parameters_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()


class Toolset:
    """The complete set of tools one agent may call."""

    def __init__(self, name: str, tools: Iterable[Tool[Any, Any]], *, max_calls: int = 20) -> None:
        self.name = name
        self._tools = {tool.name: tool for tool in tools}
        self.max_calls = max_calls
        self.calls: list[ToolCallRecord] = []

    @property
    def tools(self) -> tuple[Tool[Any, Any], ...]:
        return tuple(self._tools.values())

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def invoke(self, name: str, arguments: str | Mapping[str, Any]) -> ToolResult:
        started = time.perf_counter()
        raw = arguments if isinstance(arguments, str) else json.dumps(dict(arguments), default=str)
        result = self._run(name, arguments)

        record = ToolCallRecord(
            toolset=self.name, tool=name, arguments=raw[:_MAX_LOGGED_ARGUMENTS], ok=result.ok, error=result.error,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        self.calls.append(record)
        if result.ok:
            logger.info("tool %s.%s(%s) -> ok", self.name, name, record.arguments)
        else:
            logger.warning("tool %s.%s(%s) -> refused: %s", self.name, name, record.arguments, result.error)
        return result

    def _run(self, name: str, arguments: str | Mapping[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "none"
            return ToolResult(tool=name, ok=False,
                              error=f"Tool {name!r} is not available to the {self.name} agent (available: {available}).")
        if len(self.calls) >= self.max_calls:
            return ToolResult(tool=name, ok=False, error=f"Tool call budget of {self.max_calls} exhausted.")

        try:
            if isinstance(arguments, str):
                args = tool.args_model.model_validate_json(arguments)
            else:
                args = tool.args_model.model_validate(dict(arguments))
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors())
            return ToolResult(tool=name, ok=False, error=f"Invalid arguments for {name}: {problems}")

        try:
            data = tool.handler(args).model_dump(mode="json")
        except ToolRefused as exc:
            return ToolResult(tool=name, ok=False, error=str(exc))
        except Exception as exc:  # a tool failure (or a non-model return value) must not crash the agent loop
            logger.exception("tool %s.%s raised", self.name, name)
            return ToolResult(tool=name, ok=False, error=f"{name} failed: {type(exc).__name__}: {exc}")
        return ToolResult(tool=name, ok=True, data=data)
