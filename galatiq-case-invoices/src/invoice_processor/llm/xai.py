"""xAI Grok provider using the official ``xai-sdk`` (structured outputs via ``chat.parse``).

Every network request (``chat.parse`` / ``chat.sample``) goes through
``XAIClient._request``, which logs it before and after and records an
``LLMCallRecord`` (see ``llm/telemetry.py``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, TypeVar

import grpc
from pydantic import ValidationError
from xai_sdk import Client
from xai_sdk.chat import system as system_message
from xai_sdk.chat import tool as tool_definition
from xai_sdk.chat import tool_result
from xai_sdk.chat import user as user_message

from invoice_processor.llm.base import LLMError, LLMSettings, T
from invoice_processor.llm.telemetry import describe_error, observe_request
from invoice_processor.tools import Toolset

logger = logging.getLogger(__name__)

R = TypeVar("R")
PROVIDER = "xai"
API_KEY_ENV = "XAI_API_KEY"
_FINAL_ANSWER_PROMPT = "Stop calling tools now and give your final answer in the required structure."


class XAIClient:
    """``LLMClient`` backed by xAI Grok."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        env: Mapping[str, str] | None = None,
        sdk_client: Any = None,
    ) -> None:
        """
        Args:
            settings: Model and timeout; defaults to ``LLMSettings.from_env()``.
            env: Environment to read ``XAI_API_KEY`` from (defaults to ``os.environ``).
            sdk_client: Pre-built ``xai_sdk.Client``; for tests.

        Raises:
            LLMError: If ``XAI_API_KEY`` is not set.
        """
        self.settings = settings or LLMSettings.from_env(env)
        if sdk_client is None:
            api_key = (os.environ if env is None else env).get(API_KEY_ENV, "").strip()
            if not api_key:
                raise LLMError(f"{API_KEY_ENV} is not set; export it to use the xAI provider.")
            sdk_client = Client(api_key=api_key, timeout=self.settings.timeout_seconds)
        self._client = sdk_client

    def __repr__(self) -> str:  # never expose credentials
        return f"XAIClient(model={self.settings.model!r})"

    def complete_structured(self, system: str, prompt: str, schema: type[T], *, agent: str = "unspecified") -> T:
        chat = self._client.chat.create(model=self.settings.model)
        chat.append(system_message(system))
        chat.append(user_message(prompt))
        with self._translate_errors(schema):
            _, parsed = self._request(lambda: chat.parse(schema), agent, "structured", schema)
        return parsed

    def run_tools(
        self, system: str, prompt: str, toolset: Toolset, schema: type[T], *,
        max_rounds: int = 6, agent: str | None = None,
    ) -> T:
        agent = agent or toolset.name
        definitions = [tool_definition(name=t.name, description=t.description, parameters=t.parameters_schema())
                       for t in toolset.tools]
        chat = self._client.chat.create(model=self.settings.model, tools=definitions)
        chat.append(system_message(system))
        chat.append(user_message(prompt))

        with self._translate_errors(schema):
            for _ in range(max_rounds):
                response = self._request(chat.sample, agent, "tool_round", None)
                chat.append(response)
                if not response.tool_calls:
                    break
                for call in response.tool_calls:
                    # The only path from the model to a tool: allowlisted, validated, logged.
                    result = toolset.invoke(call.function.name, call.function.arguments)
                    chat.append(tool_result(result.model_dump_json(), tool_call_id=call.id))
            else:
                raise LLMError(f"{toolset.name} agent exceeded {max_rounds} tool-calling rounds")

            chat.append(user_message(_FINAL_ANSWER_PROMPT))
            _, parsed = self._request(lambda: chat.parse(schema), agent, "final_structured", schema)
        return parsed

    def _request(self, call: Callable[[], R], agent: str, operation: str, schema: type[Any] | None) -> R:
        """Every network request to xAI goes through here, so each one is logged and recorded."""
        return observe_request(
            call,
            agent=agent, provider=PROVIDER, model=self.settings.model, operation=operation,
            schema_name=schema.__name__ if schema else None,
            is_schema_error=lambda exc: isinstance(exc, (ValidationError, ValueError)),
            describe_response=_describe_response,
            structured=schema is not None,
        )

    @contextmanager
    def _translate_errors(self, schema: type[Any]) -> Iterator[None]:
        try:
            yield
        except grpc.RpcError as exc:
            raise LLMError(f"xAI request failed ({describe_error(exc)})") from exc
        except (ValidationError, ValueError) as exc:
            # describe_error omits the model's output, which may echo invoice contents.
            raise LLMError(f"xAI response did not match {schema.__name__}: {describe_error(exc)}") from exc


def _describe_response(result: Any) -> dict[str, Any]:
    """Request id, token usage, and cost from an SDK response (``parse`` returns (response, parsed))."""
    response = result[0] if isinstance(result, tuple) else result
    usage = getattr(response, "usage", None)

    def count(name: str) -> int | None:
        value = getattr(usage, name, None)
        return value if isinstance(value, int) else None

    tool_calls = getattr(response, "tool_calls", None)
    cost = getattr(response, "cost_usd", None)
    return {
        "request_id": getattr(response, "id", None) or None,
        "prompt_tokens": count("prompt_tokens"),
        "completion_tokens": count("completion_tokens"),
        "reasoning_tokens": count("reasoning_tokens"),
        "total_tokens": count("total_tokens"),
        "cost_usd": cost if isinstance(cost, (int, float)) else None,
        "tool_calls": len(tool_calls) if tool_calls is not None else None,
    }
