"""Offline tests for the LLM module (no network; the SDK client is faked)."""

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


def test_schema_mismatch_becomes_llm_error() -> None:
    client = XAIClient(LLMSettings(), sdk_client=FakeSDK({"result": "not a number"}))
    with pytest.raises(LLMError, match="did not match Answer"):
        client.complete_structured("s", "p", Answer)
    # Sanity: the fake really raises a ValidationError underneath.
    with pytest.raises(ValidationError):
        Answer.model_validate({"result": "not a number"})
