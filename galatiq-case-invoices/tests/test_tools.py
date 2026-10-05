"""Tool framework, per-agent toolsets, and payment authorization."""

import json
import logging
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel

from invoice_processor.agent_tools import build_payment_tools, build_validation_tools
from invoice_processor.database import init_db
from invoice_processor.inventory import SQLiteInventory
from invoice_processor.models import (
    ApprovalDecision,
    ApprovalResult,
    Invoice,
    InvoiceItem,
    IssueCode,
    PaymentResult,
    PaymentStatus,
    Severity,
    ValidationIssue,
    ValidationResult,
)
from invoice_processor.payment import (
    PaymentAuthorization,
    PaymentNotAuthorizedError,
    authorize_payment,
    mock_payment,
)
from invoice_processor.tools import Tool, ToolArgs, ToolRefused, Toolset


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return init_db(tmp_path / "inventory.db")


@pytest.fixture
def validation_tools(db_path: Path) -> Toolset:
    return build_validation_tools(SQLiteInventory(db_path))


class SpyPayment:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, vendor, amount, *, currency="USD", invoice_number=None) -> PaymentResult:
        self.calls.append((vendor, amount, currency, invoice_number))
        return mock_payment(vendor, amount, currency=currency, invoice_number=invoice_number)


AUTH = PaymentAuthorization(invoice_number="INV-1001", vendor="Widgets Inc.", amount=Decimal("5000.00"), currency="USD")


# --------------------------------------------------------------------------- #
# Generic Toolset guarantees
# --------------------------------------------------------------------------- #


class EchoArgs(ToolArgs):
    text: str


class Echo(BaseModel):
    text: str


def _echo_toolset(**kwargs) -> Toolset:
    def handler(args: EchoArgs) -> Echo:
        if args.text == "refuse":
            raise ToolRefused("Not allowed.")
        if args.text == "boom":
            raise RuntimeError("internal failure")
        return Echo(text=args.text)

    return Toolset("test", [Tool("echo", "Echo text.", EchoArgs, handler)], **kwargs)


def test_successful_call_returns_structured_data() -> None:
    result = _echo_toolset().invoke("echo", '{"text": "hi"}')
    assert (result.ok, result.data, result.error) == (True, {"text": "hi"}, None)


@pytest.mark.parametrize("name", ["__import__", "exec", "eval", "execute_sql", "mock_payment", ""])
def test_tools_outside_the_allowlist_are_refused(name: str) -> None:
    result = _echo_toolset().invoke(name, "{}")
    assert not result.ok
    assert "not available to the test agent" in result.error


@pytest.mark.parametrize(
    "arguments",
    ['not json', '[1, 2]', '{}', '{"text": 5}', '{"text": "hi", "extra": "field"}', '{"text": "hi", "__class__": "x"}'],
)
def test_invalid_arguments_are_refused(arguments: str) -> None:
    result = _echo_toolset().invoke("echo", arguments)
    assert not result.ok
    assert result.error.startswith("Invalid arguments for echo")


def test_handler_refusal_and_failure_become_error_results() -> None:
    toolset = _echo_toolset()
    assert toolset.invoke("echo", {"text": "refuse"}).error == "Not allowed."
    assert "RuntimeError: internal failure" in toolset.invoke("echo", {"text": "boom"}).error


def test_call_budget_is_enforced() -> None:
    toolset = _echo_toolset(max_calls=2)
    assert toolset.invoke("echo", {"text": "a"}).ok
    assert toolset.invoke("echo", {"text": "b"}).ok
    third = toolset.invoke("echo", {"text": "c"})
    assert not third.ok and "budget" in third.error


def test_every_call_is_recorded_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    toolset = _echo_toolset()
    with caplog.at_level(logging.INFO, logger="invoice_processor.tools"):
        toolset.invoke("echo", {"text": "hi"})
        toolset.invoke("rm_rf", {"path": "/"})

    assert [(c.tool, c.ok) for c in toolset.calls] == [("echo", True), ("rm_rf", False)]
    assert json.loads(toolset.calls[0].arguments) == {"text": "hi"}
    assert all(c.duration_ms >= 0 for c in toolset.calls)
    messages = [r.getMessage() for r in caplog.records]
    assert any("test.echo" in m and "-> ok" in m for m in messages)
    assert any("test.rm_rf" in m and "refused" in m for m in messages)


# --------------------------------------------------------------------------- #
# Validation agent toolset
# --------------------------------------------------------------------------- #


def test_validation_toolset_contains_only_inventory_tools(validation_tools: Toolset) -> None:
    assert validation_tools.names == {"lookup_inventory", "check_stock"}


def test_lookup_inventory(validation_tools: Toolset) -> None:
    assert validation_tools.invoke("lookup_inventory", {"item": "GadgetX"}).data == {
        "item": "GadgetX", "found": True, "stock": 5}
    assert validation_tools.invoke("lookup_inventory", {"item": "WidgetC"}).data == {
        "item": "WidgetC", "found": False, "stock": None}


@pytest.mark.parametrize(
    ("item", "quantity", "sufficient", "phrase"),
    [("WidgetA", 15, True, "within"), ("GadgetX", 20, False, "exceeds"),
     ("FakeItem", 1, False, "zero stock"), ("SuperGizmo", 1, False, "not in inventory")],
)
def test_check_stock(validation_tools: Toolset, item: str, quantity: int, sufficient: bool, phrase: str) -> None:
    result = validation_tools.invoke("check_stock", {"item": item, "quantity": quantity})
    assert result.ok
    assert result.data["sufficient"] is sufficient
    assert phrase in result.data["explanation"]


@pytest.mark.parametrize("args", [{"item": "WidgetA", "quantity": 0}, {"item": "WidgetA", "quantity": -5},
                                  {"item": "", "quantity": 1}, {"item": "x" * 101, "quantity": 1},
                                  {"item": "WidgetA", "quantity": 2.5}])
def test_check_stock_rejects_invalid_arguments(validation_tools: Toolset, args: dict) -> None:
    assert not validation_tools.invoke("check_stock", args).ok


def test_inventory_tools_cannot_modify_sqlite(validation_tools: Toolset, db_path: Path) -> None:
    injection = "WidgetA'; DROP TABLE inventory; --"
    result = validation_tools.invoke("lookup_inventory", {"item": injection})
    assert result.ok and result.data["found"] is False

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM inventory").fetchone()[0] == 4
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Payment agent toolset
# --------------------------------------------------------------------------- #


def test_payment_toolset_contains_only_mock_payment() -> None:
    assert build_payment_tools(AUTH).names == {"mock_payment"}


def test_payment_tools_require_an_authorization() -> None:
    with pytest.raises(TypeError, match="PaymentAuthorization"):
        build_payment_tools({"vendor": "Widgets Inc.", "amount": "5000.00"})  # type: ignore[arg-type]


def test_payment_tool_pays_the_authorized_payment_once() -> None:
    spy = SpyPayment()
    tools = build_payment_tools(AUTH, spy)

    first = tools.invoke("mock_payment", {"vendor": "Widgets Inc.", "amount": "5000"})
    second = tools.invoke("mock_payment", {"vendor": "Widgets Inc.", "amount": "5000.00"})

    assert first.ok and first.data["status"] == "paid"
    assert not second.ok and "already been paid" in second.error
    assert spy.calls == [("Widgets Inc.", Decimal("5000.00"), "USD", "INV-1001")]
    assert tools.completed_payment.status is PaymentStatus.PAID


@pytest.mark.parametrize(
    "args",
    [{"vendor": "Widgets Inc.", "amount": "5000.01"}, {"vendor": "Widgets Inc.", "amount": "50000.00"},
     {"vendor": "Evil Corp", "amount": "5000.00"}, {"vendor": "widgets inc.", "amount": "5000.00"},
     {"vendor": "Widgets Inc.", "amount": "-5000.00"}, {"vendor": "Widgets Inc.", "amount": "4999.999"},
     {"vendor": "Widgets Inc.", "amount": "5000.00", "invoice_number": "INV-9999"}],
)
def test_payment_tool_refuses_anything_but_the_approved_payment(args: dict) -> None:
    spy = SpyPayment()
    result = build_payment_tools(AUTH, spy).invoke("mock_payment", args)
    assert not result.ok
    assert spy.calls == []


# --------------------------------------------------------------------------- #
# authorize_payment
# --------------------------------------------------------------------------- #


def _invoice(number: str = "INV-1001") -> Invoice:
    return Invoice(invoice_number=number, vendor_name="Widgets Inc.", total=Decimal("5000.00"),
                   items=[InvoiceItem(name="WidgetA", quantity=10, unit_price=Decimal("500"))])


def _approval(decision: ApprovalDecision, number: str = "INV-1001") -> ApprovalResult:
    return ApprovalResult(invoice_number=number, decision=decision, reasoning="Test decision.",
                          reviewed_vendor="Widgets Inc.", reviewed_amount=Decimal("5000.00"), reviewed_currency="USD")


def test_authorize_payment_for_approved_valid_invoice() -> None:
    auth = authorize_payment(_invoice(), ValidationResult(invoice_number="INV-1001"),
                             _approval(ApprovalDecision.APPROVED))
    assert auth == AUTH


@pytest.mark.parametrize(
    ("approval", "valid", "match"),
    [(None, True, "No approval result"),
     (_approval(ApprovalDecision.REJECTED), True, "not approved"),
     (_approval(ApprovalDecision.APPROVED, "INV-9999"), True, "different invoices"),
     (_approval(ApprovalDecision.APPROVED), False, "Validation failed")],
)
def test_authorize_payment_refuses(approval, valid: bool, match: str) -> None:
    issues = [] if valid else [ValidationIssue(code=IssueCode.UNKNOWN_ITEM, severity=Severity.ERROR,
                                               message="WidgetC is not in the inventory database.")]
    with pytest.raises(PaymentNotAuthorizedError, match=match):
        authorize_payment(_invoice(), ValidationResult(invoice_number="INV-1001", issues=issues), approval)
