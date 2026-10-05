"""xAI Grok provider using the official ``xai-sdk`` (structured outputs via ``chat.parse``)."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

import grpc
from pydantic import ValidationError
from xai_sdk import Client
from xai_sdk.chat import system as system_message
from xai_sdk.chat import user as user_message

from invoice_processor.llm.base import LLMError, LLMSettings, T

logger = logging.getLogger(__name__)

API_KEY_ENV = "XAI_API_KEY"


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
        try:
            response, parsed = chat.parse(schema)
        except grpc.RpcError as exc:
            code = exc.code().name if callable(getattr(exc, "code", None)) else "UNKNOWN"
            details = exc.details() if callable(getattr(exc, "details", None)) else str(exc)
            raise LLMError(f"xAI request failed ({code}): {details}") from exc
        except (ValidationError, ValueError) as exc:
            raise LLMError(f"xAI response did not match {schema.__name__}: {exc}") from exc

        usage = getattr(response, "usage", None)
        logger.info(
            "xAI %s -> %s (prompt_tokens=%s, completion_tokens=%s, cost_usd=%s)",
            self.settings.model, schema.__name__,
            getattr(usage, "prompt_tokens", "?"), getattr(usage, "completion_tokens", "?"),
            getattr(response, "cost_usd", "?"),
        )
        return parsed
