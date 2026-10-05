"""Live smoke test against the real xAI API.

Skipped by default so the normal test suite stays offline, free, and
deterministic. Run explicitly with:

    RUN_LIVE_LLM_TESTS=1 python -m pytest -m live
"""

import os
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from invoice_processor.database import init_db
from invoice_processor.inventory import SQLiteInventory
from invoice_processor.llm import create_llm_client
from invoice_processor.models import PaymentResult
from invoice_processor.payment import mock_payment
from invoice_processor.workflow import PipelineStatus, build_workflow, run_invoice

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_LLM_TESTS") != "1", reason="set RUN_LIVE_LLM_TESTS=1 to call xAI"),
    pytest.mark.skipif(not os.environ.get("XAI_API_KEY"), reason="XAI_API_KEY is not set"),
]


class Arithmetic(BaseModel):
    result: int = Field(description="The numeric answer")


INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"


class SpyPayment:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, vendor, amount, *, currency="USD", invoice_number=None) -> PaymentResult:
        self.calls.append((vendor, amount, currency, invoice_number))
        return mock_payment(vendor, amount, currency=currency, invoice_number=invoice_number)


def test_grok_agents_with_tools(tmp_path: Path) -> None:
    spy = SpyPayment()
    workflow = build_workflow(SQLiteInventory(init_db(tmp_path / "inventory.db")), pay=spy, llm=create_llm_client())

    approved = run_invoice(INVOICES / "invoice_1001.txt", workflow)
    assert approved["status"] is PipelineStatus.COMPLETED
    assert spy.calls == [("Widgets Inc.", Decimal("5000.00"), "USD", "INV-1001")]
    assert {c.toolset for c in approved["tool_calls"]} == {"validation", "payment"}
    assert all(c.tool in {"lookup_inventory", "check_stock"} for c in approved["tool_calls"] if c.toolset == "validation")

    rejected = run_invoice(INVOICES / "invoice_1002.txt", workflow)
    assert rejected["status"] is PipelineStatus.REJECTED
    assert len(spy.calls) == 1  # no new payment
    assert "tool_calls" not in rejected or not rejected["tool_calls"]


def test_grok_returns_structured_output() -> None:
    client = create_llm_client()
    answer = client.complete_structured(
        "You are a precise calculator. Respond only with the requested structure.",
        "What is 17 + 25?",
        Arithmetic,
    )
    assert answer == Arithmetic(result=42)
