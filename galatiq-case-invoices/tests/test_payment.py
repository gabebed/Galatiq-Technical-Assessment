from decimal import Decimal
from pathlib import Path

import pytest

from invoice_processor.approval import approve_invoice
from invoice_processor.database import init_db
from invoice_processor.ingestion import extract_invoice
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
from invoice_processor.payment import mock_payment, process_payment
from invoice_processor.validation import validate_invoice


class SpyPayment:
    """Records calls and delegates to mock_payment."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, vendor, amount, *, currency="USD", invoice_number=None) -> PaymentResult:
        self.calls.append((vendor, amount, currency, invoice_number))
        return mock_payment(vendor, amount, currency=currency, invoice_number=invoice_number)


@pytest.fixture
def spy() -> SpyPayment:
    return SpyPayment()


def _invoice(total: str | None = "5000.00", vendor: str | None = "Widgets Inc.", number: str = "INV-1001") -> Invoice:
    return Invoice(
        invoice_number=number, vendor_name=vendor,
        items=[InvoiceItem(name="WidgetA", quantity=10, unit_price=Decimal("500"))],
        total=Decimal(total) if total is not None else None,
    )


def _valid(number: str = "INV-1001") -> ValidationResult:
    return ValidationResult(invoice_number=number)


def _invalid(number: str = "INV-1001") -> ValidationResult:
    error = ValidationIssue(code=IssueCode.INSUFFICIENT_STOCK, severity=Severity.ERROR,
                            message="Invoice bills 20 units of GadgetX, but only 5 are in stock.", item="GadgetX")
    return ValidationResult(invoice_number=number, issues=[error])


def _approval(decision: ApprovalDecision, number: str = "INV-1001", reasoning: str = "Within policy.") -> ApprovalResult:
    return ApprovalResult(invoice_number=number, decision=decision, reasoning=reasoning)


# --------------------------------------------------------------------------- #
# Required safety proofs
# --------------------------------------------------------------------------- #


def test_approved_invoice_calls_payment(spy: SpyPayment) -> None:
    result = process_payment(_invoice(), _valid(), _approval(ApprovalDecision.APPROVED), pay=spy)

    assert spy.calls == [("Widgets Inc.", Decimal("5000.00"), "USD", "INV-1001")]
    assert result.status is PaymentStatus.PAID
    assert result.transaction_id.startswith("MOCK-")


def test_rejected_invoice_never_calls_payment(spy: SpyPayment) -> None:
    approval = _approval(ApprovalDecision.REJECTED, reasoning="Unresolved warnings need manual review.")
    result = process_payment(_invoice(), _valid(), approval, pay=spy)

    assert spy.calls == []
    assert result.status is PaymentStatus.SKIPPED
    assert "rejected" in result.message and "Unresolved warnings need manual review." in result.message


def test_validation_failure_never_calls_payment(spy: SpyPayment) -> None:
    validation = _invalid()
    result = process_payment(_invoice(), validation, approve_invoice(_invoice(), validation), pay=spy)

    assert spy.calls == []
    assert result.status is PaymentStatus.SKIPPED


def test_validation_failure_blocks_payment_even_if_approval_says_approved(spy: SpyPayment) -> None:
    # An inconsistent (e.g. forged or buggy) approval must not be trusted on its own.
    result = process_payment(_invoice(), _invalid(), _approval(ApprovalDecision.APPROVED), pay=spy)

    assert spy.calls == []
    assert result.status is PaymentStatus.SKIPPED
    assert "Validation failed" in result.message


# --------------------------------------------------------------------------- #
# Additional gate checks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("total", ["-250.00", "0", None])
def test_non_positive_or_missing_amount_never_calls_payment(spy: SpyPayment, total: str | None) -> None:
    result = process_payment(_invoice(total=total), _valid(), _approval(ApprovalDecision.APPROVED), pay=spy)
    assert spy.calls == []
    assert result.status is PaymentStatus.SKIPPED


def test_missing_vendor_never_calls_payment(spy: SpyPayment) -> None:
    result = process_payment(_invoice(vendor=None), _valid(), _approval(ApprovalDecision.APPROVED), pay=spy)
    assert spy.calls == []
    assert "no vendor" in result.message


def test_mismatched_invoice_numbers_never_call_payment(spy: SpyPayment) -> None:
    # Approval for a different invoice must not authorize this one.
    result = process_payment(_invoice(), _valid(), _approval(ApprovalDecision.APPROVED, number="INV-9999"), pay=spy)
    assert spy.calls == []
    assert "different invoices" in result.message


def test_payment_tool_exception_returns_failed() -> None:
    def broken(*args, **kwargs) -> PaymentResult:
        raise ConnectionError("bank offline")

    result = process_payment(_invoice(), _valid(), _approval(ApprovalDecision.APPROVED), pay=broken)
    assert result.status is PaymentStatus.FAILED
    assert "bank offline" in result.message


# --------------------------------------------------------------------------- #
# mock_payment tool
# --------------------------------------------------------------------------- #


def test_mock_payment_returns_structured_result() -> None:
    result = mock_payment("Widgets Inc.", Decimal("5000.00"))
    assert result.status is PaymentStatus.PAID
    assert (result.vendor_name, result.amount, result.currency) == ("Widgets Inc.", Decimal("5000.00"), "USD")
    assert result.message == "Paid 5,000.00 USD to Widgets Inc."


def test_mock_payment_transaction_ids_are_unique() -> None:
    ids = {mock_payment("V", Decimal("1")).transaction_id for _ in range(50)}
    assert len(ids) == 50


@pytest.mark.parametrize(("vendor", "amount"), [("", Decimal("10")), ("V", Decimal("0")), ("V", Decimal("-5")), ("V", 10.0)])
def test_mock_payment_refuses_invalid_input(vendor: str, amount: object) -> None:
    result = mock_payment(vendor, amount)
    assert result.status is PaymentStatus.FAILED
    assert result.transaction_id is None


# --------------------------------------------------------------------------- #
# Duplicate payment guard
# --------------------------------------------------------------------------- #


def test_duplicate_guard_blocks_second_payment_for_same_invoice() -> None:
    from invoice_processor.payment import DuplicatePaymentGuard

    spy = SpyPayment()
    guard = DuplicatePaymentGuard(spy)
    first = guard("Precision Parts Ltd.", Decimal("1890.00"), invoice_number="INV-1004")
    revised = guard("Precision Parts Ltd.", Decimal("5940.00"), invoice_number="INV-1004")
    other = guard("Widgets Inc.", Decimal("5000.00"), invoice_number="INV-1001")

    assert [r.status for r in (first, revised, other)] == [PaymentStatus.PAID, PaymentStatus.FAILED, PaymentStatus.PAID]
    assert len(spy.calls) == 2
    assert first.transaction_id in revised.message


def test_duplicate_guard_allows_retry_after_failed_payment() -> None:
    from invoice_processor.payment import DuplicatePaymentGuard

    guard = DuplicatePaymentGuard(mock_payment)
    assert guard("", Decimal("10"), invoice_number="INV-1").status is PaymentStatus.FAILED
    assert guard("Vendor", Decimal("10"), invoice_number="INV-1").status is PaymentStatus.PAID


# --------------------------------------------------------------------------- #
# Integration: payment is called for exactly the approved sample invoices
# --------------------------------------------------------------------------- #

INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"


def test_full_pipeline_pays_only_approved_samples(tmp_path: Path) -> None:
    inventory = SQLiteInventory(init_db(tmp_path / "inventory.db"))
    spy = SpyPayment()
    paid, approved = set(), set()

    for path in sorted(INVOICES.iterdir()):
        invoice = extract_invoice(path)
        validation = validate_invoice(invoice, inventory)
        approval = approve_invoice(invoice, validation)
        calls_before = len(spy.calls)
        result = process_payment(invoice, validation, approval, pay=spy)

        if approval.is_approved:
            approved.add(path.name)
        if result.status is PaymentStatus.PAID:
            paid.add(path.name)
        if not validation.is_valid or not approval.is_approved:
            assert len(spy.calls) == calls_before, f"{path.name} reached the payment tool"

    assert paid == approved
    assert "invoice_1002.txt" not in paid and "invoice_1003.txt" not in paid
    assert "invoice_1001.txt" in paid
