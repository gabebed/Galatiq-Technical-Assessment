"""Payment stage: a local mock payment tool and the gate that guards it.

``process_payment`` is the only sanctioned way to pay an invoice. It calls the
payment function only when the invoice was explicitly APPROVED *and* passed
validation, re-checking both instead of trusting upstream stages. Anything else
yields a SKIPPED result explaining why. Everything is local; no banking API.
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal
from typing import Protocol

from invoice_processor.approval import hard_failures
from invoice_processor.models import ApprovalResult, Invoice, PaymentResult, PaymentStatus, ValidationResult

logger = logging.getLogger(__name__)


class PaymentFunction(Protocol):
    def __call__(
        self, vendor: str, amount: Decimal, *, currency: str = "USD", invoice_number: str | None = None
    ) -> PaymentResult: ...


def mock_payment(
    vendor: str, amount: Decimal, *, currency: str = "USD", invoice_number: str | None = None
) -> PaymentResult:
    """Simulate a bank transfer. Never raises: invalid input returns a FAILED result."""
    vendor = (vendor or "").strip()
    problems = []
    if not vendor:
        problems.append("no vendor given")
    if not isinstance(amount, Decimal) or amount <= 0:
        problems.append(f"amount must be a positive Decimal, got {amount!r}")
    if problems:
        logger.warning("Mock payment refused for %s: %s", invoice_number, "; ".join(problems))
        return PaymentResult(
            invoice_number=invoice_number, status=PaymentStatus.FAILED, vendor_name=vendor or None,
            currency=currency, message="Payment refused: " + "; ".join(problems),
        )

    transaction_id = f"MOCK-{uuid.uuid4().hex[:12].upper()}"
    logger.info("Paid %s %s to %s (invoice %s, txn %s)", f"{amount:,.2f}", currency, vendor, invoice_number, transaction_id)
    return PaymentResult(
        invoice_number=invoice_number, status=PaymentStatus.PAID, vendor_name=vendor, amount=amount,
        currency=currency, transaction_id=transaction_id, message=f"Paid {amount:,.2f} {currency} to {vendor}",
    )


def payment_blockers(invoice: Invoice, validation: ValidationResult, approval: ApprovalResult) -> list[str]:
    """Every reason this invoice must not be paid. Empty means payment may proceed."""
    blockers = []
    if not approval.is_approved:
        blockers.append(f"Invoice was {approval.decision.value}, not approved. Approver reasoning: {approval.reasoning}")
    if not validation.is_valid:
        blockers.append(f"Validation failed with {len(validation.errors)} error(s).")
    ids = {invoice.invoice_number, validation.invoice_number, approval.invoice_number}
    if len(ids) != 1:
        blockers.append(f"Invoice, validation, and approval refer to different invoices: {sorted(map(str, ids))}.")
    blockers.extend(r for r in hard_failures(invoice, validation) if r not in {e.message for e in validation.errors})
    if invoice.vendor_name is None:
        blockers.append("Invoice has no vendor to pay.")
    return blockers


def process_payment(
    invoice: Invoice,
    validation: ValidationResult,
    approval: ApprovalResult,
    pay: PaymentFunction = mock_payment,
) -> PaymentResult:
    """Pay an approved, valid invoice; otherwise return SKIPPED without calling ``pay``."""
    blockers = payment_blockers(invoice, validation, approval)
    if blockers:
        logger.info("Payment skipped for %s: %s", invoice.invoice_number, " | ".join(blockers))
        return PaymentResult(
            invoice_number=invoice.invoice_number, status=PaymentStatus.SKIPPED, vendor_name=invoice.vendor_name,
            amount=invoice.total, currency=invoice.currency, message="Payment not made. " + " ".join(blockers),
        )

    try:
        return pay(invoice.vendor_name, invoice.total, currency=invoice.currency, invoice_number=invoice.invoice_number)
    except Exception as exc:  # a failing payment tool must not crash the pipeline
        logger.exception("Payment tool failed for %s", invoice.invoice_number)
        return PaymentResult(
            invoice_number=invoice.invoice_number, status=PaymentStatus.FAILED, vendor_name=invoice.vendor_name,
            amount=invoice.total, currency=invoice.currency, message=f"Payment tool error: {exc}",
        )
