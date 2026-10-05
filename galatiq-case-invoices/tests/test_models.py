from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

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

# --------------------------------------------------------------------------- #
# Invoice / InvoiceItem
# --------------------------------------------------------------------------- #


def test_clean_invoice_parses_from_json_like_data() -> None:  # INV-1004
    invoice = Invoice.model_validate(
        {
            "invoice_number": "INV-1004",
            "vendor_name": "Precision Parts Ltd.",
            "invoice_date": "2026-01-22",
            "due_date": "2026-02-22",
            "items": [
                {"name": "WidgetA", "quantity": 3, "unit_price": 250.00},
                {"name": "WidgetB", "quantity": 2, "unit_price": 500.00},
            ],
            "subtotal": "1750.00",
            "tax_rate": 0.08,
            "tax_amount": 140.00,
            "total": 1890.00,
        }
    )

    assert invoice.due_date == date(2026, 2, 22)
    assert invoice.currency == "USD"
    assert invoice.total == Decimal("1890")
    assert invoice.computed_subtotal == Decimal("1750")


def test_invalid_document_data_is_representable() -> None:  # INV-1009
    invoice = Invoice(
        invoice_number="INV-1009",
        vendor_name="",
        vendor_address=None,
        due_date=None,
        payment_terms="   ",
        items=[
            InvoiceItem(name="WidgetA", quantity=-5, unit_price=Decimal("250")),
            InvoiceItem(name="WidgetB", quantity=2, unit_price=Decimal("500")),
        ],
        subtotal=Decimal("1000"),
        total=Decimal("-250"),
    )

    assert invoice.vendor_name is None
    assert invoice.payment_terms is None
    assert invoice.items[0].quantity == -5
    assert invoice.computed_subtotal == Decimal("-250") != invoice.subtotal


def test_item_quantities_combines_repeated_lines() -> None:  # INV-1013
    lines = [("WidgetA", 15), ("WidgetB", 10), ("GadgetX", 5), ("WidgetA", 5),
             ("WidgetB", 8), ("GadgetX", 3), ("WidgetA", 2), ("GadgetX", 1)]
    invoice = Invoice(items=[InvoiceItem(name=n, quantity=q) for n, q in lines])

    assert invoice.item_quantities() == {"WidgetA": 22, "WidgetB": 18, "GadgetX": 9}


def test_computed_subtotal_detects_stated_total_error() -> None:  # INV-1007
    invoice = Invoice(
        items=[
            InvoiceItem(name="WidgetA", quantity=20, unit_price=Decimal("250.00")),
            InvoiceItem(name="WidgetB", quantity=15, unit_price=Decimal("500.00")),
            InvoiceItem(name="GadgetX", quantity=3, unit_price=Decimal("750.00")),
        ],
        tax_amount=Decimal("885.00"),
        total=Decimal("15525.00"),
    )

    assert invoice.computed_subtotal + invoice.tax_amount == Decimal("15635.00")
    assert invoice.computed_subtotal + invoice.tax_amount != invoice.total


def test_computed_subtotal_is_none_when_a_line_is_incomplete() -> None:
    invoice = Invoice(items=[InvoiceItem(name="WidgetA", quantity=1)])
    assert invoice.items[0].computed_total is None
    assert invoice.computed_subtotal is None
    assert Invoice().computed_subtotal is None


def test_currency_is_normalized_and_validated() -> None:  # INV-1014
    assert Invoice(currency=" eur ").currency == "EUR"
    with pytest.raises(ValidationError):
        Invoice(currency="EURO")


def test_unparseable_date_is_rejected() -> None:  # INV-1003 "yesterday"
    with pytest.raises(ValidationError):
        Invoice(due_date="yesterday")


@pytest.mark.parametrize("name", ["", "   "])
def test_item_requires_a_name(name: str) -> None:
    with pytest.raises(ValidationError):
        InvoiceItem(name=name, quantity=1)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Invoice(vendor="Widgets Inc.")


def test_models_are_immutable() -> None:
    invoice = Invoice(invoice_number="INV-1001")
    with pytest.raises(ValidationError):
        invoice.invoice_number = "INV-9999"


def test_invoice_json_round_trip_preserves_decimals() -> None:
    invoice = Invoice(
        invoice_number="INV-1013",
        invoice_date=date(2026, 1, 24),
        items=[InvoiceItem(name="WidgetA", quantity=5, unit_price=Decimal("240.00"), note="Volume discount")],
        tax_amount=Decimal("1472.80"),
    )
    assert Invoice.model_validate_json(invoice.model_dump_json()) == invoice


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _issue(severity: Severity) -> ValidationIssue:
    return ValidationIssue(
        code=IssueCode.INSUFFICIENT_STOCK,
        severity=severity,
        message="Requested quantity exceeds stock",
        item="GadgetX",
        expected="<= 5",
        actual="20",
    )


def test_validation_result_with_no_issues_is_valid() -> None:
    assert ValidationResult(invoice_number="INV-1001").is_valid


def test_warnings_do_not_invalidate() -> None:
    result = ValidationResult(issues=[_issue(Severity.WARNING)])
    assert result.is_valid
    assert len(result.warnings) == 1 and not result.errors


def test_errors_invalidate() -> None:
    result = ValidationResult(issues=[_issue(Severity.ERROR), _issue(Severity.WARNING)])
    assert not result.is_valid
    assert [i.severity for i in result.errors] == [Severity.ERROR]
    assert result.model_dump()["is_valid"] is False


def test_issue_requires_message_and_known_code() -> None:
    with pytest.raises(ValidationError):
        ValidationIssue(code=IssueCode.UNKNOWN_ITEM, severity=Severity.ERROR, message=" ")
    with pytest.raises(ValidationError):
        ValidationIssue(code="not_a_code", severity=Severity.ERROR, message="x")


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #


def test_approval_result() -> None:
    result = ApprovalResult(
        invoice_number="INV-1002",
        decision="rejected",
        reasoning="Quantity exceeds stock; total above $10K.",
        requires_additional_scrutiny=True,
        flags=["over_10k"],
    )
    assert result.decision is ApprovalDecision.REJECTED
    assert not result.is_approved


def test_approval_requires_reasoning() -> None:
    with pytest.raises(ValidationError):
        ApprovalResult(decision=ApprovalDecision.APPROVED, reasoning="")


# --------------------------------------------------------------------------- #
# Payment
# --------------------------------------------------------------------------- #


def test_paid_result() -> None:
    result = PaymentResult(
        invoice_number="INV-1001",
        status=PaymentStatus.PAID,
        vendor_name="Widgets Inc.",
        amount=Decimal("5000.00"),
    )
    assert result.status is PaymentStatus.PAID
    assert result.processed_at.tzinfo is not None


@pytest.mark.parametrize(
    "overrides",
    [{"vendor_name": None}, {"amount": None}, {"amount": Decimal("0")}, {"amount": Decimal("-250")}],
)
def test_paid_result_requires_vendor_and_positive_amount(overrides: dict) -> None:
    data = {"status": PaymentStatus.PAID, "vendor_name": "Widgets Inc.", "amount": Decimal("10")} | overrides
    with pytest.raises(ValidationError):
        PaymentResult(**data)


def test_skipped_payment_needs_no_amount() -> None:
    result = PaymentResult(invoice_number="INV-1003", status="skipped", message="Rejected at approval")
    assert result.amount is None
