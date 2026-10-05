from decimal import Decimal
from pathlib import Path

import pytest

from invoice_processor.approval import (
    FLAG_BLOCKING_WARNINGS,
    FLAG_CURRENCY_NOT_CONVERTED,
    FLAG_INVALID_AMOUNT,
    FLAG_OVER_THRESHOLD,
    FLAG_OVERRIDDEN,
    FLAG_VALIDATION_FAILED,
    ApprovalPolicy,
    approve_invoice,
)
from invoice_processor.database import init_db
from invoice_processor.ingestion import extract_invoice
from invoice_processor.inventory import SQLiteInventory
from invoice_processor.models import (
    ApprovalDecision,
    ApprovalResult,
    Invoice,
    InvoiceItem,
    IssueCode,
    Severity,
    ValidationIssue,
    ValidationResult,
)
from invoice_processor.validation import validate_invoice

APPROVED, REJECTED = ApprovalDecision.APPROVED, ApprovalDecision.REJECTED


def _invoice(total: str | None, currency: str = "USD") -> Invoice:
    return Invoice(
        invoice_number="INV-T1",
        vendor_name="Widgets Inc.",
        currency=currency,
        items=[InvoiceItem(name="WidgetA", quantity=1, unit_price=Decimal(total or "1"))],
        total=Decimal(total) if total is not None else None,
    )


def _validation(*issues: ValidationIssue) -> ValidationResult:
    return ValidationResult(invoice_number="INV-T1", issues=list(issues))


def _issue(code: IssueCode, severity: Severity, message: str = "Something needs attention here.") -> ValidationIssue:
    return ValidationIssue(code=code, severity=severity, message=message)


# --------------------------------------------------------------------------- #
# Required scenarios
# --------------------------------------------------------------------------- #


def test_valid_invoice_below_threshold_is_approved() -> None:
    result = approve_invoice(_invoice("5000.00"), _validation())

    assert result.decision is APPROVED
    assert result.requires_additional_scrutiny is False
    assert "$5,000.00 is within the $10,000.00 threshold" in result.reasoning
    assert result.flags == []


def test_valid_invoice_above_threshold_is_approved_with_scrutiny() -> None:
    result = approve_invoice(_invoice("15000.00"), _validation())

    assert result.decision is APPROVED
    assert result.requires_additional_scrutiny is True
    assert FLAG_OVER_THRESHOLD in result.flags
    assert "exceeds the $10,000.00 threshold" in result.reasoning


def test_invoice_with_validation_failures_is_rejected() -> None:
    error = _issue(IssueCode.INSUFFICIENT_STOCK, Severity.ERROR, "Invoice bills 20 units of GadgetX, but only 5 are in stock.")
    result = approve_invoice(_invoice("5000.00"), _validation(error))

    assert result.decision is REJECTED
    assert FLAG_VALIDATION_FAILED in result.flags
    assert error.message in result.reasoning


@pytest.mark.parametrize("total", ["-250.00", "0", None])
def test_invalid_amounts_are_rejected_even_if_validation_passed(total: str | None) -> None:
    # A "clean" ValidationResult simulates validation being skipped or incomplete.
    result = approve_invoice(_invoice(total), _validation())

    assert result.decision is REJECTED
    assert FLAG_INVALID_AMOUNT in result.flags
    # A missing total cannot be shown to be under the threshold.
    assert result.requires_additional_scrutiny is (total is None)


def test_exactly_threshold_does_not_require_scrutiny() -> None:
    result = approve_invoice(_invoice("10000.00"), _validation())
    assert result.decision is APPROVED
    assert result.requires_additional_scrutiny is False


def test_one_cent_over_threshold_requires_scrutiny() -> None:
    result = approve_invoice(_invoice("10000.01"), _validation())
    assert result.decision is APPROVED
    assert result.requires_additional_scrutiny is True


# --------------------------------------------------------------------------- #
# Warnings and scrutiny
# --------------------------------------------------------------------------- #


def test_warnings_below_threshold_are_noted_but_do_not_block() -> None:
    warning = _issue(IssueCode.EXTRACTION_UNCERTAIN, Severity.WARNING, "Corrected OCR artifact in line total.")
    result = approve_invoice(_invoice("9975.00"), _validation(warning))

    assert result.decision is APPROVED
    assert "warning:extraction_uncertain" in result.flags
    assert warning.message in result.reasoning


def test_warnings_above_threshold_block_approval() -> None:
    warning = _issue(IssueCode.EXTRACTION_UNCERTAIN, Severity.WARNING)
    result = approve_invoice(_invoice("15000.00"), _validation(warning))

    assert result.decision is REJECTED
    assert result.requires_additional_scrutiny is True
    assert FLAG_BLOCKING_WARNINGS in result.flags


def test_suspicious_content_blocks_at_any_amount() -> None:
    warning = _issue(IssueCode.SUSPICIOUS_CONTENT, Severity.WARNING, "Invoice uses pressure language (urgent).")
    result = approve_invoice(_invoice("100.00"), _validation(warning))
    assert result.decision is REJECTED
    assert FLAG_BLOCKING_WARNINGS in result.flags


def test_unconvertible_currency_requires_scrutiny() -> None:
    warning = _issue(IssueCode.UNSUPPORTED_CURRENCY, Severity.WARNING)
    result = approve_invoice(_invoice("4125.00", "EUR"), _validation(warning))

    assert result.decision is APPROVED  # the currency warning itself is exempt
    assert result.requires_additional_scrutiny is True
    assert FLAG_CURRENCY_NOT_CONVERTED in result.flags


def test_configured_exchange_rate_is_used_for_threshold() -> None:
    policy = ApprovalPolicy(usd_exchange_rates={"USD": Decimal(1), "EUR": Decimal("1.10")})
    over = approve_invoice(_invoice("9500.00", "EUR"), _validation(), policy=policy)  # $10,450
    under = approve_invoice(_invoice("9000.00", "EUR"), _validation(), policy=policy)  # $9,900

    assert over.requires_additional_scrutiny and "$10,450.00 exceeds" in over.reasoning
    assert not under.requires_additional_scrutiny


def test_custom_threshold() -> None:
    policy = ApprovalPolicy(scrutiny_threshold_usd=Decimal("1000"))
    assert approve_invoice(_invoice("5000.00"), _validation(), policy=policy).requires_additional_scrutiny


# --------------------------------------------------------------------------- #
# Hard-rule enforcement for pluggable approvers
# --------------------------------------------------------------------------- #


class RubberStampApprover:
    """Stand-in for a future LLM approver that misbehaves."""

    def review(self, invoice: Invoice, validation: ValidationResult) -> ApprovalResult:
        return ApprovalResult(invoice_number=invoice.invoice_number, decision=APPROVED,
                              reasoning="Looks fine to me.", reviewer="rubber-stamp")


def test_custom_approver_cannot_approve_hard_failures() -> None:
    error = _issue(IssueCode.UNKNOWN_ITEM, Severity.ERROR, "WidgetC is not in the inventory database.")
    result = approve_invoice(_invoice("5000.00"), _validation(error), approver=RubberStampApprover())

    assert result.decision is REJECTED
    assert FLAG_OVERRIDDEN in result.flags
    assert "Looks fine to me." in result.reasoning and error.message in result.reasoning


def test_custom_approver_cannot_skip_scrutiny() -> None:
    result = approve_invoice(_invoice("50000.00"), _validation(), approver=RubberStampApprover())
    assert result.decision is APPROVED
    assert result.requires_additional_scrutiny is True


def test_custom_approver_decision_is_kept_when_valid() -> None:
    result = approve_invoice(_invoice("5000.00"), _validation(), approver=RubberStampApprover())
    assert result.reviewer == "rubber-stamp"
    assert result.reasoning == "Looks fine to me."


def test_mismatched_validation_result_is_rejected() -> None:
    with pytest.raises(ValueError, match="INV-OTHER"):
        approve_invoice(_invoice("5000.00"), ValidationResult(invoice_number="INV-OTHER"))


# --------------------------------------------------------------------------- #
# Integration: every sample invoice through ingestion -> validation -> approval
# --------------------------------------------------------------------------- #

INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"
EXPECTED_APPROVED = {
    "invoice_1001.txt", "invoice_1004.json", "invoice_1004_revised.json", "invoice_1006.csv",
    "invoice_1010.txt", "invoice_1011.txt", "invoice_1011.pdf", "invoice_1012.txt", "invoice_1012.pdf",
    "invoice_1014.xml", "invoice_1015.csv",
}


@pytest.fixture(scope="module")
def inventory(tmp_path_factory: pytest.TempPathFactory) -> SQLiteInventory:
    return SQLiteInventory(init_db(tmp_path_factory.mktemp("db") / "inventory.db"))


@pytest.mark.parametrize("path", sorted(INVOICES.iterdir()), ids=lambda p: p.name)
def test_sample_invoices_end_to_end(path: Path, inventory: SQLiteInventory) -> None:
    invoice = extract_invoice(path)
    validation = validate_invoice(invoice, inventory)
    result = approve_invoice(invoice, validation)

    expected = APPROVED if path.name in EXPECTED_APPROVED else REJECTED
    assert result.decision is expected
    if not validation.is_valid:
        assert result.decision is REJECTED
    # Scrutiny: every invoice over $10K, plus EUR invoice 1014 (no FX rate configured).
    over = invoice.total is None or invoice.currency != "USD" or invoice.total > 10000
    assert result.requires_additional_scrutiny is over
