"""Payment stage: a local mock payment tool and the gate that guards it.

``process_payment`` is the only sanctioned way to pay an invoice. It calls the
payment function only when the invoice was explicitly APPROVED *and* passed
validation, re-checking both instead of trusting upstream stages. Anything else
yields a SKIPPED result explaining why. Everything is local; no banking API.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
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
    if not isinstance(amount, Decimal) or not amount.is_finite() or amount <= 0:
        problems.append(f"amount must be a positive, finite Decimal, got {amount!r}")
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


class DuplicatePaymentGuard:
    """``PaymentFunction`` wrapper that refuses to pay an invoice number twice.

    Covers re-submitted or revised invoices (e.g. INV-1004 and its revision R1)
    within one process. State is in memory only; a persistent ledger would be
    needed across runs.
    """

    def __init__(self, pay: PaymentFunction = mock_payment) -> None:
        self._pay = pay
        self._paid: dict[str, PaymentResult] = {}

    def __call__(
        self, vendor: str, amount: Decimal, *, currency: str = "USD", invoice_number: str | None = None
    ) -> PaymentResult:
        previous = self._paid.get(invoice_number) if invoice_number else None
        if previous is not None:
            logger.warning("Duplicate payment blocked for %s (already paid in %s)",
                           invoice_number, previous.transaction_id)
            return PaymentResult(
                invoice_number=invoice_number, status=PaymentStatus.FAILED, vendor_name=vendor, amount=amount,
                currency=currency,
                message=(f"Duplicate payment blocked: invoice {invoice_number} was already paid "
                         f"{previous.amount:,.2f} {previous.currency} (transaction {previous.transaction_id}). "
                         "A revised or resubmitted invoice needs manual reconciliation."),
            )
        result = self._pay(vendor, amount, currency=currency, invoice_number=invoice_number)
        if invoice_number and result.status is PaymentStatus.PAID:
            self._paid[invoice_number] = result
        return result


def payment_blockers(
    invoice: Invoice, validation: ValidationResult, approval: ApprovalResult | None
) -> list[str]:
    """Every reason this invoice must not be paid. Empty means payment may proceed."""
    blockers = []
    if approval is None:
        blockers.append("No approval result: payment requires an explicit approval.")
    elif not approval.is_approved:
        blockers.append(f"Invoice was {approval.decision.value}, not approved. Approver reasoning: {approval.reasoning}")
    elif not approval.covers_terms(invoice):
        blockers.append(
            f"The approval does not cover this invoice's terms: approved {approval.reviewed_amount} "
            f"{approval.reviewed_currency} to {approval.reviewed_vendor!r}, but the invoice is "
            f"{invoice.total} {invoice.currency} to {invoice.vendor_name!r}."
        )
    if not validation.is_valid:
        blockers.append(f"Validation failed with {len(validation.errors)} error(s).")
    ids = {invoice.invoice_number, validation.invoice_number}
    if approval is not None:
        ids.add(approval.invoice_number)
    if len(ids) != 1:
        blockers.append(f"Invoice, validation, and approval refer to different invoices: {sorted(map(str, ids))}.")
    if invoice.invoice_number is None:
        blockers.append("Invoice has no invoice number to record the payment against.")
    blockers.extend(r for r in hard_failures(invoice, validation) if r not in {e.message for e in validation.errors})
    if invoice.vendor_name is None:
        blockers.append("Invoice has no vendor to pay.")
    return blockers


class PaymentNotAuthorizedError(Exception):
    def __init__(self, blockers: list[str]) -> None:
        super().__init__(" ".join(blockers))
        self.blockers = blockers


@dataclass(frozen=True)
class PaymentAuthorization:
    """Proof that one specific payment passed validation and approval.

    Obtain only via ``authorize_payment``. Payment tools are built from an
    authorization and will pay exactly this vendor and amount, nothing else.
    """

    invoice_number: str
    vendor: str
    amount: Decimal
    currency: str


def authorize_payment(
    invoice: Invoice, validation: ValidationResult, approval: ApprovalResult | None
) -> PaymentAuthorization:
    """Raises ``PaymentNotAuthorizedError`` unless the invoice is approved and valid."""
    blockers = payment_blockers(invoice, validation, approval)
    if blockers:
        raise PaymentNotAuthorizedError(blockers)
    assert invoice.invoice_number and invoice.vendor_name and invoice.total  # guaranteed by payment_blockers
    return PaymentAuthorization(
        invoice_number=invoice.invoice_number, vendor=invoice.vendor_name, amount=invoice.total,
        currency=invoice.currency,
    )


def skipped_payment(invoice: Invoice, blockers: list[str]) -> PaymentResult:
    logger.info("Payment skipped for %s: %s", invoice.invoice_number, " | ".join(blockers))
    return PaymentResult(
        invoice_number=invoice.invoice_number, status=PaymentStatus.SKIPPED, vendor_name=invoice.vendor_name,
        amount=invoice.total, currency=invoice.currency, message="Payment not made. " + " ".join(blockers),
    )


def execute_authorized_payment(authorization: PaymentAuthorization, pay: PaymentFunction) -> PaymentResult:
    """Call the payment function for an authorization; payment-tool errors become FAILED results."""
    try:
        return pay(authorization.vendor, authorization.amount, currency=authorization.currency,
                   invoice_number=authorization.invoice_number)
    except Exception as exc:  # a failing payment tool must not crash the pipeline
        logger.exception("Payment tool failed for %s", authorization.invoice_number)
        return PaymentResult(
            invoice_number=authorization.invoice_number, status=PaymentStatus.FAILED,
            vendor_name=authorization.vendor, amount=authorization.amount, currency=authorization.currency,
            message=f"Payment tool error: {exc}",
        )


def process_payment(
    invoice: Invoice,
    validation: ValidationResult,
    approval: ApprovalResult | None,
    pay: PaymentFunction = mock_payment,
) -> PaymentResult:
    """Pay an approved, valid invoice; otherwise return SKIPPED without calling ``pay``."""
    try:
        authorization = authorize_payment(invoice, validation, approval)
    except PaymentNotAuthorizedError as exc:
        return skipped_payment(invoice, exc.blockers)
    return execute_authorized_payment(authorization, pay)
