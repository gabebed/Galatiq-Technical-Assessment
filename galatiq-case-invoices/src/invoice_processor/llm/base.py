"""Provider-agnostic LLM interface and configuration.

The rest of the application depends only on ``LLMClient``. The LLM returns
structured reasoning (Pydantic models); it never touches the database or the
payment tool, which stay deterministic.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

from invoice_processor.tools import Toolset

T = TypeVar("T", bound=BaseModel)

DEFAULT_PROVIDER = "xai"
DEFAULT_MODEL = "grok-4.7"
DEFAULT_TIMEOUT_SECONDS = 60.0


class LLMError(RuntimeError):
    """Raised when the LLM is misconfigured, unreachable, or returns unusable output."""


class LLMClient(Protocol):
    def complete_structured(self, system: str, prompt: str, schema: type[T]) -> T:
        """Send a system + user prompt and return a validated instance of ``schema``."""
        ...

    def run_tools(self, system: str, prompt: str, toolset: Toolset, schema: type[T], *, max_rounds: int = 6) -> T:
        """Let the model call tools from ``toolset`` (only via ``Toolset.invoke``), then
        return its final answer as a validated instance of ``schema``."""
        ...


@dataclass(frozen=True)
class LLMSettings:
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMSettings:
        """Read LLM_PROVIDER, LLM_MODEL, LLM_TIMEOUT_SECONDS (API keys are read by each provider)."""
        env = os.environ if env is None else env
        try:
            timeout = float(env.get("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
        except ValueError as exc:
            raise LLMError(f"LLM_TIMEOUT_SECONDS must be a number, got {env['LLM_TIMEOUT_SECONDS']!r}") from exc
        if timeout <= 0:
            raise LLMError("LLM_TIMEOUT_SECONDS must be positive")
        return cls(
            provider=env.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER,
            model=env.get("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            timeout_seconds=timeout,
        )
