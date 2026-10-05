"""xAI Grok provider using the official ``xai-sdk`` (structured outputs via ``chat.parse``)."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import grpc
from pydantic import ValidationError
from xai_sdk import Client
from xai_sdk.chat import system as system_message
from xai_sdk.chat import tool as tool_definition
from xai_sdk.chat import tool_result
from xai_sdk.chat import user as user_message

from invoice_processor.llm.base import LLMError, LLMSettings, T
from invoice_processor.tools import Toolset

logger = logging.getLogger(__name__)

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

    def complete_structured(self, system: str, prompt: str, schema: type[T]) -> T:
        chat = self._client.chat.create(model=self.settings.model)
        chat.append(system_message(system))
        chat.append(user_message(prompt))
        with self._translate_errors(schema):
            response, parsed = chat.parse(schema)
        self._log_usage(schema, response)
        return parsed

    def run_tools(self, system: str, prompt: str, toolset: Toolset, schema: type[T], *, max_rounds: int = 6) -> T:
        definitions = [tool_definition(name=t.name, description=t.description, parameters=t.parameters_schema())
                       for t in toolset.tools]
        chat = self._client.chat.create(model=self.settings.model, tools=definitions)
        chat.append(system_message(system))
        chat.append(user_message(prompt))

        with self._translate_errors(schema):
            for _ in range(max_rounds):
                response = chat.sample()
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
            response, parsed = chat.parse(schema)
        self._log_usage(schema, response)
        return parsed

    @contextmanager
    def _translate_errors(self, schema: type[Any]) -> Iterator[None]:
        try:
            yield
        except grpc.RpcError as exc:
            code = exc.code().name if callable(getattr(exc, "code", None)) else "UNKNOWN"
            details = exc.details() if callable(getattr(exc, "details", None)) else str(exc)
            raise LLMError(f"xAI request failed ({code}): {details}") from exc
        except (ValidationError, ValueError) as exc:
            raise LLMError(f"xAI response did not match {schema.__name__}: {exc}") from exc

    def _log_usage(self, schema: type[Any], response: Any) -> None:
        usage = getattr(response, "usage", None)
        logger.info(
            "xAI %s -> %s (prompt_tokens=%s, completion_tokens=%s, cost_usd=%s)",
            self.settings.model, schema.__name__,
            getattr(usage, "prompt_tokens", "?"), getattr(usage, "completion_tokens", "?"),
            getattr(response, "cost_usd", "?"),
        )
