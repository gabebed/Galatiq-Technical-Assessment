"""Live smoke test against the real xAI API.

Skipped by default so the normal test suite stays offline, free, and
deterministic. Run explicitly with:

    RUN_LIVE_LLM_TESTS=1 python -m pytest -m live
"""

import os

import pytest
from pydantic import BaseModel, Field

from invoice_processor.llm import create_llm_client

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_LLM_TESTS") != "1", reason="set RUN_LIVE_LLM_TESTS=1 to call xAI"),
    pytest.mark.skipif(not os.environ.get("XAI_API_KEY"), reason="XAI_API_KEY is not set"),
]


class Arithmetic(BaseModel):
    result: int = Field(description="The numeric answer")


def test_grok_returns_structured_output() -> None:
    client = create_llm_client()
    answer = client.complete_structured(
        "You are a precise calculator. Respond only with the requested structure.",
        "What is 17 + 25?",
        Arithmetic,
    )
    assert answer == Arithmetic(result=42)
