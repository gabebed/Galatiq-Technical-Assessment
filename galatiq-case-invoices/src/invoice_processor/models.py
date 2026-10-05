"""Typed data contracts shared by every pipeline stage.

    Invoice -> ValidationResult -> ApprovalResult -> PaymentResult

Design rule: ``Invoice`` records what the document *says*, not whether it is
acceptable. Negative quantities, empty vendors, or totals that do not add up
are representable so the validation stage can report them as
``ValidationIssue``s instead of extraction crashing. The models only enforce
types and shape; business rules live in the validation stage.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    model_validator,
)


def _blank_to_none(value: Any) -> Any:
    """Treat empty or whitespace-only strings as missing data."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _upper(value: Any) -> Any:
    return value.strip().upper() if isinstance(value, str) else value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


OptionalText = Annotated[str | None, BeforeValidator(_blank_to_none)]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CurrencyCode = Annotated[str, BeforeValidator(_upper), StringConstraints(pattern=r"^[A-Z]{3}$")]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


class InvoiceItem(_Model):
    """A single line on an invoice, as stated by the vendor."""

    name: NonEmptyText = Field(description="Item name, normalized to the inventory key where possible.")
    quantity: int | None = Field(default=None, description="Signed; negatives are flagged by validation.")
    unit_price: Decimal | None = None
    line_total: Decimal | None = Field(default=None, description="Line amount as printed on the invoice.")
    note: OptionalText = Field(default=None, description="Qualifier such as 'Volume discount' or 'rush order'.")

    @property
    def computed_total(self) -> Decimal | None:
        """quantity * unit_price, or None if either is missing."""
        if self.quantity is None or self.unit_price is None:
            return None
        return self.quantity * self.unit_price


class Invoice(_Model):
    """Structured invoice data extracted from a source document."""

    invoice_number: OptionalText = None
    revision: OptionalText = None
    vendor_name: OptionalText = None
    vendor_address: OptionalText = None
    invoice_date: date | None = None
    due_date: date | None = None
    currency: CurrencyCode = "USD"
    items: list[InvoiceItem] = Field(default_factory=list)

    # Amounts as stated on the document; compare against computed_subtotal.
    subtotal: Decimal | None = None
    tax_rate: Decimal | None = Field(default=None, description="Fraction, e.g. 0.08 for 8%.")
    tax_amount: Decimal | None = None
    shipping: Decimal | None = None
    total: Decimal | None = None

    payment_terms: OptionalText = None
    notes: OptionalText = None
    source_file: OptionalText = Field(default=None, description="Path of the document this was extracted from.")
    extraction_warnings: list[NonEmptyText] = Field(
        default_factory=list,
        description="Values that were unparseable or corrected during ingestion.",
    )

    @property
    def computed_subtotal(self) -> Decimal | None:
        """Sum of line quantity * unit_price, or None if any line is incomplete."""
        totals = [item.computed_total for item in self.items]
        if not totals or any(t is None for t in totals):
            return None
        return sum(totals, Decimal("0"))

    def item_quantities(self) -> dict[str, int]:
        """Total quantity per item name, combining repeated lines.

        Lines with an unknown quantity are skipped.
        """
        quantities: dict[str, int] = defaultdict(int)
        for item in self.items:
            if item.quantity is not None:
                quantities[item.name] += item.quantity
        return dict(quantities)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class Severity(str, Enum):
    ERROR = "error"  # blocks approval
    WARNING = "warning"  # needs human/VP attention but is not disqualifying


class IssueCode(str, Enum):
    MISSING_FIELD = "missing_field"
    INVALID_DATE = "invalid_date"
    INVALID_QUANTITY = "invalid_quantity"
    INVALID_AMOUNT = "invalid_amount"
    AMOUNT_MISMATCH = "amount_mismatch"
    UNKNOWN_ITEM = "unknown_item"
    OUT_OF_STOCK = "out_of_stock"
    INSUFFICIENT_STOCK = "insufficient_stock"
    DUPLICATE_INVOICE = "duplicate_invoice"
    UNSUPPORTED_CURRENCY = "unsupported_currency"
    SUSPICIOUS_CONTENT = "suspicious_content"


class ValidationIssue(_Model):
    code: IssueCode
    severity: Severity
    message: NonEmptyText
    field: OptionalText = Field(default=None, description="Offending field, e.g. 'due_date' or 'items[0].quantity'.")
    item: OptionalText = Field(default=None, description="Inventory item the issue concerns, if any.")
    expected: OptionalText = None
    actual: OptionalText = None


class ValidationResult(_Model):
    invoice_number: OptionalText = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=_utc_now)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_valid(self) -> bool:
        """True when there are no ERROR-severity issues."""
        return not self.errors

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalResult(_Model):
    invoice_number: OptionalText = None
    decision: ApprovalDecision
    reasoning: NonEmptyText = Field(description="Why the decision was made; logged for every outcome.")
    requires_additional_scrutiny: bool = Field(default=False, description="e.g. total above the $10K threshold.")
    flags: list[NonEmptyText] = Field(default_factory=list, description="Policy rules that fired.")
    reviewer: NonEmptyText = "vp-approval"
    decided_at: datetime = Field(default_factory=_utc_now)

    @property
    def is_approved(self) -> bool:
        return self.decision is ApprovalDecision.APPROVED


# --------------------------------------------------------------------------- #
# Payment
# --------------------------------------------------------------------------- #


class PaymentStatus(str, Enum):
    PAID = "paid"
    FAILED = "failed"
    SKIPPED = "skipped"  # invoice was not approved


class PaymentResult(_Model):
    invoice_number: OptionalText = None
    status: PaymentStatus
    vendor_name: OptionalText = None
    amount: Decimal | None = None
    currency: CurrencyCode = "USD"
    transaction_id: OptionalText = None
    message: OptionalText = None
    processed_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _paid_requires_payee_and_amount(self) -> PaymentResult:
        if self.status is PaymentStatus.PAID:
            if self.vendor_name is None:
                raise ValueError("a PAID result requires vendor_name")
            if self.amount is None or self.amount <= 0:
                raise ValueError("a PAID result requires a positive amount")
        return self
