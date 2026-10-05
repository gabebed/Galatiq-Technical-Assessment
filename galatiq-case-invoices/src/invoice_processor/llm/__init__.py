"""LLM access. Use ``create_llm_client()``; select the provider with LLM_PROVIDER."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from invoice_processor.llm.base import LLMClient, LLMError, LLMSettings
from invoice_processor.llm.xai import XAIClient

# provider name -> factory(settings, env). Add new providers here.
PROVIDERS: dict[str, Callable[[LLMSettings, Mapping[str, str] | None], LLMClient]] = {
    "xai": lambda settings, env: XAIClient(settings, env=env),
}


def create_llm_client(settings: LLMSettings | None = None, env: Mapping[str, str] | None = None) -> LLMClient:
    """Build the configured LLM client.

    Raises:
        LLMError: If the provider is unknown or its credentials are missing.
    """
    settings = settings or LLMSettings.from_env(env)
    factory = PROVIDERS.get(settings.provider)
    if factory is None:
        raise LLMError(f"Unknown LLM provider {settings.provider!r}; available: {', '.join(sorted(PROVIDERS))}")
    return factory(settings, env)


__all__ = ["PROVIDERS", "LLMClient", "LLMError", "LLMSettings", "XAIClient", "create_llm_client"]
