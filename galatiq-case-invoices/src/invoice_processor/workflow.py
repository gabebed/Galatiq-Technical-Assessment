"""LangGraph workflow wiring the existing stages together.

    START -> ingest -> validate -> approve -> pay -> END
                |          |          |
                |          +--------> reject -> END
                v          v
               END        END          (failed: unreadable file / inventory unavailable)

Nodes are thin adapters: all business rules stay in ingestion, validation,
approval, and payment. Inventory access and the payment function are injected
explicitly.

With an ``llm``, two agents participate, each limited to its own toolset:
the validation agent (read-only inventory tools, advisory output only) and the
payment agent (``mock_payment`` bound to a ``PaymentAuthorization``). Without
an ``llm`` the workflow is fully deterministic.
"""

from __future__ import annotations

import logging
import operator
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from invoice_processor.agents import PaymentReport, ValidationReview, review_validation, run_payment_agent
from invoice_processor.approval import ApprovalPolicy, Approver, approve_invoice
from invoice_processor.database import InventoryDatabaseError
from invoice_processor.ingestion import IngestionError, extract_invoice
from invoice_processor.inventory import InventoryLookup
from invoice_processor.llm import LLMClient, LLMError
from invoice_processor.models import (
    ApprovalResult,
    Invoice,
    PaymentResult,
    PaymentStatus,
    ValidationResult,
)
from invoice_processor.payment import (
    PaymentFunction,
    PaymentNotAuthorizedError,
    authorize_payment,
    execute_authorized_payment,
    mock_payment,
    skipped_payment,
)
from invoice_processor.tools import ToolCallRecord
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
    validation_review: ValidationReview  # advisory, from the validation agent
    approval: ApprovalResult
    payment: PaymentResult
    payment_report: PaymentReport  # from the payment agent
    status: PipelineStatus
    rejection_reason: str
    error: str
    tool_calls: Annotated[list[ToolCallRecord], operator.add]  # audit trail across agents
    agent_errors: Annotated[list[str], operator.add]


def _route_after_ingest(state: InvoiceState) -> Literal["validate", "__end__"]:
    return "validate" if "invoice" in state else END


def _route_after_validate(state: InvoiceState) -> Literal["approve", "reject", "__end__"]:
    if "validation" not in state:
        return END
    return "approve" if state["validation"].is_valid else "reject"


def _route_after_approve(state: InvoiceState) -> Literal["pay", "reject"]:
    return "pay" if state["approval"].is_approved else "reject"


def payment_step(
    invoice: Invoice,
    validation: ValidationResult,
    approval: ApprovalResult | None,
    *,
    pay: PaymentFunction = mock_payment,
    llm: LLMClient | None = None,
) -> InvoiceState:
    """Authorize, then pay deterministically or via the payment agent.

    Routing already guarantees approval, but this step re-checks: without a
    ``PaymentAuthorization`` neither ``pay`` nor the payment tool can be reached.
    """
    try:
        authorization = authorize_payment(invoice, validation, approval)
    except PaymentNotAuthorizedError as exc:
        logger.error("Payment step reached without authorization for %s", invoice.invoice_number)
        return {"payment": skipped_payment(invoice, exc.blockers), "status": PipelineStatus.REJECTED,
                "rejection_reason": " ".join(exc.blockers)}

    update: InvoiceState = {}
    if llm is None:
        result = execute_authorized_payment(authorization, pay)
    else:
        result, report, calls = run_payment_agent(llm, authorization, pay)
        update["tool_calls"] = calls
        if report is not None:
            update["payment_report"] = report

    update["payment"] = result
    update["status"] = PipelineStatus.COMPLETED if result.status is PaymentStatus.PAID else PipelineStatus.FAILED
    if update["status"] is PipelineStatus.FAILED:
        update["error"] = result.message or "Payment did not complete"
    return update


def build_workflow(
    inventory: InventoryLookup,
    *,
    pay: PaymentFunction = mock_payment,
    approver: Approver | None = None,
    policy: ApprovalPolicy | None = None,
    llm: LLMClient | None = None,
) -> CompiledStateGraph:
    """Compile the invoice-processing graph with the given tools (and optional LLM agents)."""

    def ingest(state: InvoiceState) -> InvoiceState:
        try:
            return {"invoice": extract_invoice(state["invoice_path"])}
        except IngestionError as exc:
            logger.error("Ingestion failed: %s", exc)
            return {"status": PipelineStatus.FAILED, "error": str(exc)}

    def validate(state: InvoiceState) -> InvoiceState:
        invoice = state["invoice"]
        try:
            validation = validate_invoice(invoice, inventory)
        except InventoryDatabaseError as exc:  # fail closed: never continue unvalidated
            logger.error("Validation could not run for %s: %s", invoice.invoice_number, exc)
            return {"status": PipelineStatus.FAILED, "error": str(exc)}

        update: InvoiceState = {"validation": validation}
        if llm is not None and validation.is_valid:
            try:
                review, calls = review_validation(llm, invoice, validation, inventory)
                update["validation_review"] = review
                update["tool_calls"] = calls
            except LLMError as exc:  # advisory: deterministic result stands
                logger.warning("Validation agent unavailable for %s: %s", invoice.invoice_number, exc)
                update["agent_errors"] = [f"validation agent: {exc}"]
        return update

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
        return payment_step(state["invoice"], state["validation"], state.get("approval"), pay=pay, llm=llm)

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
