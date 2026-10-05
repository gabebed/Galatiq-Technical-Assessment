"""LLM agents. Each gets only its own toolset; neither can change deterministic outcomes.

* Validation agent: reviews a passing invoice using read-only inventory tools and
  returns advisory concerns. It cannot alter the ``ValidationResult``.
* Payment agent: given a ``PaymentAuthorization``, executes the approved payment
  through the bound ``mock_payment`` tool. The recorded ``PaymentResult`` is what
  the tool actually did, never the agent's claim.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from invoice_processor.agent_tools import build_payment_tools, build_validation_tools
from invoice_processor.inventory import InventoryLookup
from invoice_processor.llm import LLMClient
from invoice_processor.models import Invoice, PaymentResult, PaymentStatus, ValidationResult
from invoice_processor.payment import PaymentAuthorization, PaymentFunction, mock_payment
from invoice_processor.tools import ToolCallRecord

logger = logging.getLogger(__name__)


class AgentConcern(BaseModel):
    item: str | None = Field(default=None, description="Inventory item the concern is about, if any.")
    concern: str = Field(min_length=1, description="What looks wrong and what evidence supports it.")


class ValidationReview(BaseModel):
    summary: str = Field(min_length=1, description="One- or two-sentence assessment.")
    concerns: list[AgentConcern] = Field(default_factory=list, description="Substantiated concerns only.")


class PaymentReport(BaseModel):
    summary: str = Field(min_length=1, description="What was paid, or why payment was not made.")


VALIDATION_SYSTEM = """You are the validation agent in an invoice-processing system.
Deterministic checks have already run; their results are authoritative and you cannot change them.
Your job is to double-check line items against inventory with your tools and report any substantiated
concerns the deterministic checks may have missed (e.g. an item whose name looks like a variant of a
stocked item). Use lookup_inventory and check_stock; quantities for repeated items should be summed.
Do not speculate without tool evidence. You cannot make payments or modify inventory."""

PAYMENT_SYSTEM = """You are the payment agent in an invoice-processing system.
The invoice below has passed validation and been approved. Execute the approved payment exactly once
by calling mock_payment with the approved vendor and amount, copied exactly. Do not alter either value.
If the tool refuses, do not retry with different values; report what happened."""


def review_validation(
    llm: LLMClient, invoice: Invoice, validation: ValidationResult, inventory: InventoryLookup
) -> tuple[ValidationReview, list[ToolCallRecord]]:
    """Advisory review; raises ``LLMError`` if the model fails."""
    toolset = build_validation_tools(inventory)
    prompt = (
        f"Invoice (extracted):\n{invoice.model_dump_json(indent=2, exclude={'source_file'})}\n\n"
        f"Deterministic validation result:\n{validation.model_dump_json(indent=2)}"
    )
    review = llm.run_tools(VALIDATION_SYSTEM, prompt, toolset, ValidationReview)
    return review, list(toolset.calls)


def run_payment_agent(
    llm: LLMClient, authorization: PaymentAuthorization, pay: PaymentFunction = mock_payment
) -> tuple[PaymentResult, PaymentReport | None, list[ToolCallRecord]]:
    """Let the agent execute an authorized payment. Never raises for agent misbehaviour."""
    toolset = build_payment_tools(authorization, pay)
    prompt = (
        f"Approved payment:\n- invoice: {authorization.invoice_number}\n- vendor: {authorization.vendor}\n"
        f"- amount: {authorization.amount}\n- currency: {authorization.currency}"
    )
    report: PaymentReport | None = None
    failure = None
    try:
        report = llm.run_tools(PAYMENT_SYSTEM, prompt, toolset, PaymentReport, max_rounds=3)
    except Exception as exc:  # any failure, possibly *after* money moved: the tool's record is the truth
        failure = f"Payment agent error: {type(exc).__name__}: {exc}"
        logger.error("Payment agent failed for %s: %s", authorization.invoice_number, failure)

    # Recorded from what the tool executed, never from the agent's report.
    payment = toolset.completed_payment or (toolset.payments[-1] if toolset.payments else None)
    if payment is None:
        reason = failure or (report.summary if report else "no reason given")
        payment = PaymentResult(
            invoice_number=authorization.invoice_number, status=PaymentStatus.FAILED,
            vendor_name=authorization.vendor, amount=authorization.amount, currency=authorization.currency,
            message=f"Payment agent did not execute the payment: {reason}",
        )
    return payment, report, list(toolset.calls)
