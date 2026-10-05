"""The LangGraph workflow must reproduce the direct function pipeline exactly."""

from decimal import Decimal
from pathlib import Path

import pytest

from invoice_processor.approval import approve_invoice
from invoice_processor.database import init_db
from invoice_processor.ingestion import extract_invoice
from invoice_processor.inventory import SQLiteInventory
from invoice_processor.models import ApprovalDecision, ApprovalResult, Invoice, PaymentResult, PaymentStatus
from invoice_processor.payment import mock_payment, process_payment
from invoice_processor.validation import validate_invoice
from invoice_processor.workflow import PipelineStatus, build_workflow, run_invoice

INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"
SAMPLES = sorted(INVOICES.iterdir())


class SpyPayment:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def __call__(self, vendor, amount, *, currency="USD", invoice_number=None) -> PaymentResult:
        self.calls.append(invoice_number)
        return mock_payment(vendor, amount, currency=currency, invoice_number=invoice_number)


@pytest.fixture(scope="module")
def inventory(tmp_path_factory: pytest.TempPathFactory) -> SQLiteInventory:
    return SQLiteInventory(init_db(tmp_path_factory.mktemp("db") / "inventory.db"))


def _issues(result) -> list[tuple]:
    return [(i.code, i.severity, i.message, i.field, i.item) for i in result.issues]


# --------------------------------------------------------------------------- #
# Graph structure
# --------------------------------------------------------------------------- #


def test_graph_has_expected_nodes_and_edges(inventory: SQLiteInventory) -> None:
    graph = build_workflow(inventory).get_graph()
    assert set(graph.nodes) == {"__start__", "ingest", "validate", "approve", "reject", "pay", "__end__"}
    edges = {(e.source, e.target) for e in graph.edges}
    assert edges == {
        ("__start__", "ingest"),
        ("ingest", "validate"), ("ingest", "__end__"),
        ("validate", "approve"), ("validate", "reject"), ("validate", "__end__"),
        ("approve", "pay"), ("approve", "reject"),
        ("reject", "__end__"),
        ("pay", "__end__"),
    }


# --------------------------------------------------------------------------- #
# Equivalence with the direct function pipeline
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_workflow_matches_direct_pipeline(path: Path, inventory: SQLiteInventory) -> None:
    # Direct pipeline (as used in earlier milestones).
    invoice = extract_invoice(path)
    validation = validate_invoice(invoice, inventory)
    approval = approve_invoice(invoice, validation)
    payment = process_payment(invoice, validation, approval)

    state = run_invoice(path, build_workflow(inventory))

    assert state["invoice"] == invoice
    assert _issues(state["validation"]) == _issues(validation)
    assert state["validation"].is_valid == validation.is_valid

    if not validation.is_valid:
        # Graph short-circuits before approval; the direct pipeline rejects at approval.
        assert "approval" not in state
        assert approval.decision is ApprovalDecision.REJECTED
        assert state["status"] is PipelineStatus.REJECTED
        for error in validation.errors:
            assert error.message in state["rejection_reason"]
    else:
        assert (state["approval"].decision, state["approval"].reasoning, state["approval"].flags,
                state["approval"].requires_additional_scrutiny) == (
               approval.decision, approval.reasoning, approval.flags, approval.requires_additional_scrutiny)

    if payment.status is PaymentStatus.PAID:
        assert state["status"] is PipelineStatus.COMPLETED
        assert state["payment"].status is PaymentStatus.PAID
        assert (state["payment"].vendor_name, state["payment"].amount) == (payment.vendor_name, payment.amount)
    else:
        assert state["status"] is PipelineStatus.REJECTED
        assert "payment" not in state


def test_outcome_summary(inventory: SQLiteInventory) -> None:
    workflow = build_workflow(inventory)
    statuses = {p.name: run_invoice(p, workflow)["status"] for p in SAMPLES}
    completed = {name for name, s in statuses.items() if s is PipelineStatus.COMPLETED}
    assert completed == {
        "invoice_1001.txt", "invoice_1004.json", "invoice_1004_revised.json", "invoice_1006.csv",
        "invoice_1010.txt", "invoice_1011.txt", "invoice_1011.pdf", "invoice_1012.txt", "invoice_1012.pdf",
        "invoice_1014.xml", "invoice_1015.csv",
    }
    assert all(s is PipelineStatus.REJECTED for name, s in statuses.items() if name not in completed)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_validation_failure_routes_to_reject_without_approval_or_payment(inventory: SQLiteInventory) -> None:
    spy = SpyPayment()
    state = run_invoice(INVOICES / "invoice_1002.txt", build_workflow(inventory, pay=spy))

    assert state["status"] is PipelineStatus.REJECTED
    assert "approval" not in state and "payment" not in state
    assert "20 units of GadgetX" in state["rejection_reason"]
    assert spy.calls == []


def test_approval_rejection_routes_to_reject_without_payment(inventory: SQLiteInventory) -> None:
    class RejectAll:
        def review(self, invoice: Invoice, validation) -> ApprovalResult:
            return ApprovalResult(invoice_number=invoice.invoice_number, decision=ApprovalDecision.REJECTED,
                                  reasoning="VP declined: vendor not on approved list.")

    spy = SpyPayment()
    state = run_invoice(INVOICES / "invoice_1001.txt", build_workflow(inventory, pay=spy, approver=RejectAll()))

    assert state["status"] is PipelineStatus.REJECTED
    assert state["validation"].is_valid
    assert state["rejection_reason"] == "VP declined: vendor not on approved list."
    assert "payment" not in state
    assert spy.calls == []


def test_approved_invoice_routes_to_payment(inventory: SQLiteInventory) -> None:
    spy = SpyPayment()
    state = run_invoice(INVOICES / "invoice_1001.txt", build_workflow(inventory, pay=spy))

    assert state["status"] is PipelineStatus.COMPLETED
    assert spy.calls == ["INV-1001"]
    assert state["payment"].amount == Decimal("5000.00")
    assert "rejection_reason" not in state


def test_unreadable_file_fails_at_ingestion(tmp_path: Path, inventory: SQLiteInventory) -> None:
    state = run_invoice(tmp_path / "missing.txt", build_workflow(inventory))
    assert state["status"] is PipelineStatus.FAILED
    assert "Could not read" in state["error"]
    assert "invoice" not in state


def test_unavailable_inventory_fails_closed(tmp_path: Path) -> None:
    spy = SpyPayment()
    workflow = build_workflow(SQLiteInventory(tmp_path / "missing.db"), pay=spy)
    state = run_invoice(INVOICES / "invoice_1001.txt", workflow)

    assert state["status"] is PipelineStatus.FAILED
    assert "not found" in state["error"]
    assert "validation" not in state and "approval" not in state
    assert spy.calls == []


def test_payment_tool_failure_marks_run_failed(inventory: SQLiteInventory) -> None:
    def broken(*args, **kwargs) -> PaymentResult:
        raise ConnectionError("bank offline")

    state = run_invoice(INVOICES / "invoice_1001.txt", build_workflow(inventory, pay=broken))
    assert state["status"] is PipelineStatus.FAILED
    assert state["payment"].status is PaymentStatus.FAILED
    assert "bank offline" in state["error"]
