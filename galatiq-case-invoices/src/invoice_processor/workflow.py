"""LangGraph workflow wiring the existing stages together.

    START -> ingest -> validate -> approve -> pay -> END
                |          |          |
                |          +--------> reject -> END
                v          v
               END        END          (failed: unreadable file / inventory unavailable)

Nodes are thin adapters: all business rules stay in ingestion, validation,
approval, and payment. Inventory access and the payment function are injected
explicitly so they can be swapped (tests, or agent tools later).
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from invoice_processor.approval import ApprovalPolicy, Approver, approve_invoice
from invoice_processor.database import InventoryDatabaseError
from invoice_processor.ingestion import IngestionError, extract_invoice
from invoice_processor.inventory import InventoryLookup
from invoice_processor.models import (
    ApprovalResult,
    Invoice,
    PaymentResult,
    PaymentStatus,
    ValidationResult,
)
from invoice_processor.payment import PaymentFunction, mock_payment, process_payment
from invoice_processor.validation import validate_invoice

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    COMPLETED = "completed"  # approved and paid
    REJECTED = "rejected"  # failed validation or approval; payment never attempted
    FAILED = "failed"  # could not be processed (unreadable file, inventory down, payment error)


class InvoiceState(TypedDict, total=False):
    invoice_path: str
    invoice: Invoice
    validation: ValidationResult
    approval: ApprovalResult
    payment: PaymentResult
    status: PipelineStatus
    rejection_reason: str
    error: str


def _route_after_ingest(state: InvoiceState) -> Literal["validate", "__end__"]:
    return "validate" if "invoice" in state else END


def _route_after_validate(state: InvoiceState) -> Literal["approve", "reject", "__end__"]:
    if "validation" not in state:
        return END
    return "approve" if state["validation"].is_valid else "reject"


def _route_after_approve(state: InvoiceState) -> Literal["pay", "reject"]:
    return "pay" if state["approval"].is_approved else "reject"


def build_workflow(
    inventory: InventoryLookup,
    *,
    pay: PaymentFunction = mock_payment,
    approver: Approver | None = None,
    policy: ApprovalPolicy | None = None,
) -> CompiledStateGraph:
    """Compile the invoice-processing graph with the given tools."""

    def ingest(state: InvoiceState) -> InvoiceState:
        try:
            return {"invoice": extract_invoice(state["invoice_path"])}
        except IngestionError as exc:
            logger.error("Ingestion failed: %s", exc)
            return {"status": PipelineStatus.FAILED, "error": str(exc)}

    def validate(state: InvoiceState) -> InvoiceState:
        try:
            return {"validation": validate_invoice(state["invoice"], inventory)}
        except InventoryDatabaseError as exc:  # fail closed: never continue unvalidated
            logger.error("Validation could not run for %s: %s", state["invoice"].invoice_number, exc)
            return {"status": PipelineStatus.FAILED, "error": str(exc)}

    def approve(state: InvoiceState) -> InvoiceState:
        return {"approval": approve_invoice(state["invoice"], state["validation"], approver, policy)}

    def reject(state: InvoiceState) -> InvoiceState:
        if "approval" in state:
            reason = state["approval"].reasoning
        else:
            errors = state["validation"].errors
            reason = "\n".join([f"Rejected at validation with {len(errors)} error(s):"] + [f"- {e.message}" for e in errors])
        logger.info("Invoice %s rejected:\n%s", state["invoice"].invoice_number, reason)
        return {"status": PipelineStatus.REJECTED, "rejection_reason": reason}

    def pay_invoice(state: InvoiceState) -> InvoiceState:
        # process_payment re-checks approval and validation before calling the payment tool.
        result = process_payment(state["invoice"], state["validation"], state["approval"], pay=pay)
        status = PipelineStatus.COMPLETED if result.status is PaymentStatus.PAID else PipelineStatus.FAILED
        update: InvoiceState = {"payment": result, "status": status}
        if status is PipelineStatus.FAILED:
            update["error"] = result.message or "Payment did not complete"
        return update

    graph = StateGraph(InvoiceState)
    graph.add_node("ingest", ingest)
    graph.add_node("validate", validate)
    graph.add_node("approve", approve)
    graph.add_node("reject", reject)
    graph.add_node("pay", pay_invoice)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", _route_after_ingest)
    graph.add_conditional_edges("validate", _route_after_validate)
    graph.add_conditional_edges("approve", _route_after_approve)
    graph.add_edge("reject", END)
    graph.add_edge("pay", END)
    return graph.compile()


def run_invoice(path: str | Path, workflow: CompiledStateGraph) -> InvoiceState:
    """Process one invoice file through the compiled workflow."""
    return workflow.invoke({"invoice_path": str(path)})
