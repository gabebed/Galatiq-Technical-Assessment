"""Validation stage: deterministic checks of an ``Invoice`` against inventory and itself.

ERROR issues are hard failures: ``ValidationResult.is_valid`` is False and the
invoice must not be approved. WARNING issues are passed to approval for review.

Stock rule: quantities for repeated lines of the same item are summed and
compared against current stock. Stock is not decremented across invoices.

If the inventory cannot be queried, ``InventoryDatabaseError`` propagates: the
validator never reports an invoice as valid without checking it.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from decimal import Decimal

from invoice_processor.inventory import InventoryLookup
from invoice_processor.models import Invoice, IssueCode, Severity, ValidationIssue, ValidationResult

logger = logging.getLogger(__name__)

AMOUNT_TOLERANCE = Decimal("0.01")
_SUSPICIOUS = re.compile(r"\burgent\b|\bimmediate(?:ly)?\b|\bwire transfer\b|\bpenalt(?:y|ies)\b", re.IGNORECASE)
_NET_TERMS = re.compile(r"\bnet\s*(\d+)\b", re.IGNORECASE)
_WARNING_FIELD = re.compile(r"^([\w\[\].]+):\s")


def _error(code: IssueCode, message: str, **kwargs: str | None) -> ValidationIssue:
    return ValidationIssue(code=code, severity=Severity.ERROR, message=message, **kwargs)


def _warning(code: IssueCode, message: str, **kwargs: str | None) -> ValidationIssue:
    return ValidationIssue(code=code, severity=Severity.WARNING, message=message, **kwargs)


def _money(amount: Decimal, currency: str) -> str:
    return f"{amount:,.2f} {currency}"


def _differs(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) > AMOUNT_TOLERANCE


def _warned_fields(invoice: Invoice) -> set[str]:
    return {m.group(1) for w in invoice.extraction_warnings if (m := _WARNING_FIELD.match(w))}


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_required_fields(invoice: Invoice) -> list[ValidationIssue]:
    issues = []
    required = {
        "invoice_number": "an invoice number is needed to track the invoice and detect duplicates",
        "vendor_name": "payment cannot be made without knowing the payee",
        "total": "there is no stated amount to approve or pay",
    }
    for field, why in required.items():
        if getattr(invoice, field) is None:
            issues.append(_error(IssueCode.MISSING_FIELD, f"Missing {field.replace('_', ' ')}: {why}.", field=field))
    if not invoice.items:
        issues.append(_error(IssueCode.MISSING_FIELD, "Invoice has no line items to verify.", field="items"))

    # Dates that failed to parse are already reported via extraction warnings.
    warned = _warned_fields(invoice)
    for field in ("invoice_date", "due_date"):
        if getattr(invoice, field) is None and field not in warned:
            issues.append(
                _warning(IssueCode.MISSING_FIELD, f"Missing {field.replace('_', ' ')}; payment timing cannot be checked.", field=field)
            )
    return issues


def check_extraction_warnings(invoice: Invoice) -> list[ValidationIssue]:
    """Surface ingestion uncertainty (unparseable or OCR-corrected values) for review."""
    issues = []
    for warning in invoice.extraction_warnings:
        match = _WARNING_FIELD.match(warning)
        issues.append(
            _warning(
                IssueCode.EXTRACTION_UNCERTAIN,
                f"Extraction uncertainty - {warning}. Verify against the source document.",
                field=match.group(1) if match else None,
            )
        )
    return issues


def check_quantities(invoice: Invoice) -> list[ValidationIssue]:
    issues = []
    for index, item in enumerate(invoice.items):
        field = f"items[{index}].quantity"
        if item.quantity is None:
            issues.append(_error(
                IssueCode.INVALID_QUANTITY,
                f"Quantity for {item.name} could not be read, so stock and amounts cannot be verified.",
                field=field, item=item.name,
            ))
        elif item.quantity <= 0:
            issues.append(_error(
                IssueCode.INVALID_QUANTITY,
                f"Quantity for {item.name} is {item.quantity}; invoiced quantities must be positive.",
                field=field, item=item.name, expected="> 0", actual=str(item.quantity),
            ))
    return issues


def make_inventory_check(inventory: InventoryLookup) -> Callable[[Invoice], list[ValidationIssue]]:
    def check_inventory(invoice: Invoice) -> list[ValidationIssue]:
        names = list(dict.fromkeys(item.name for item in invoice.items))  # first-seen order
        stock = inventory.get_stock_levels(names)

        requested: dict[str, int] = defaultdict(int)
        line_counts: dict[str, int] = defaultdict(int)
        for item in invoice.items:
            if item.quantity is not None and item.quantity > 0:  # invalid lines reported separately
                requested[item.name] += item.quantity
                line_counts[item.name] += 1

        issues = []
        for name in names:
            qty = requested.get(name, 0)
            lines = f" across {line_counts[name]} lines" if line_counts[name] > 1 else ""
            if name not in stock:
                issues.append(_error(
                    IssueCode.UNKNOWN_ITEM,
                    f"{name} is not in the inventory database; the item cannot be verified as a product Acme stocks.",
                    item=name,
                ))
            elif qty and stock[name] == 0:
                issues.append(_error(
                    IssueCode.OUT_OF_STOCK,
                    f"{name} has zero stock, but the invoice bills {qty} units{lines}. "
                    "Billing for an item with no inventory is a possible fraud signal.",
                    item=name, expected="0", actual=str(qty),
                ))
            elif qty > stock[name]:
                issues.append(_error(
                    IssueCode.INSUFFICIENT_STOCK,
                    f"Invoice bills {qty} units of {name}{lines}, but only {stock[name]} are in stock.",
                    item=name, expected=f"<= {stock[name]}", actual=str(qty),
                ))
        return issues

    return check_inventory


def check_amounts(invoice: Invoice) -> list[ValidationIssue]:
    issues = []
    cur = invoice.currency

    if invoice.total is not None and invoice.total <= 0:
        issues.append(_error(
            IssueCode.INVALID_AMOUNT,
            f"Total is {_money(invoice.total, cur)}; an invoice total must be positive.",
            field="total", expected="> 0", actual=str(invoice.total),
        ))

    for index, item in enumerate(invoice.items):
        if item.unit_price is not None and item.unit_price < 0:
            issues.append(_error(
                IssueCode.INVALID_AMOUNT,
                f"Unit price for {item.name} is negative ({_money(item.unit_price, cur)}).",
                field=f"items[{index}].unit_price", item=item.name, actual=str(item.unit_price),
            ))
        computed = item.computed_total
        if computed is not None and item.line_total is not None and _differs(computed, item.line_total):
            issues.append(_error(
                IssueCode.AMOUNT_MISMATCH,
                f"Line total for {item.name} is {_money(item.line_total, cur)}, but "
                f"{item.quantity} x {_money(item.unit_price, cur)} = {_money(computed, cur)}.",
                field=f"items[{index}].line_total", item=item.name, expected=str(computed), actual=str(item.line_total),
            ))

    computed_subtotal = invoice.computed_subtotal
    if computed_subtotal is not None and invoice.subtotal is not None and _differs(computed_subtotal, invoice.subtotal):
        issues.append(_error(
            IssueCode.AMOUNT_MISMATCH,
            f"Stated subtotal {_money(invoice.subtotal, cur)} does not match the sum of line items "
            f"({_money(computed_subtotal, cur)}).",
            field="subtotal", expected=str(computed_subtotal), actual=str(invoice.subtotal),
        ))

    base = invoice.subtotal if invoice.subtotal is not None else computed_subtotal
    if base is not None and invoice.total is not None:
        tax = invoice.tax_amount or Decimal(0)
        shipping = invoice.shipping or Decimal(0)
        expected = base + tax + shipping
        if _differs(expected, invoice.total):
            parts = [f"subtotal {_money(base, cur)}"]
            if tax:
                parts.append(f"tax {_money(tax, cur)}")
            if shipping:
                parts.append(f"shipping {_money(shipping, cur)}")
            breakdown = " + ".join(parts) + (f" = {_money(expected, cur)}" if len(parts) > 1 else "")
            issues.append(_error(
                IssueCode.AMOUNT_MISMATCH,
                f"Stated total {_money(invoice.total, cur)} does not equal {breakdown} "
                f"(difference {_money(invoice.total - expected, cur)}).",
                field="total", expected=str(expected), actual=str(invoice.total),
            ))

    if base is not None and invoice.tax_rate is not None and invoice.tax_amount is not None:
        expected_tax = (base * invoice.tax_rate).quantize(Decimal("0.01"))
        if _differs(expected_tax, invoice.tax_amount):
            # Printed rates are often rounded, so this is a warning rather than an error.
            issues.append(_warning(
                IssueCode.AMOUNT_MISMATCH,
                f"Tax amount {_money(invoice.tax_amount, cur)} does not match {invoice.tax_rate:%} of "
                f"{_money(base, cur)} ({_money(expected_tax, cur)}).",
                field="tax_amount", expected=str(expected_tax), actual=str(invoice.tax_amount),
            ))
    return issues


def check_dates(invoice: Invoice) -> list[ValidationIssue]:
    if invoice.invoice_date is None or invoice.due_date is None:
        return []
    if invoice.due_date < invoice.invoice_date:
        return [_error(
            IssueCode.INVALID_DATE,
            f"Due date {invoice.due_date} is before the invoice date {invoice.invoice_date}.",
            field="due_date", expected=f">= {invoice.invoice_date}", actual=str(invoice.due_date),
        )]
    terms = _NET_TERMS.search(invoice.payment_terms or "")
    if invoice.due_date == invoice.invoice_date and terms and int(terms.group(1)) > 0:
        return [_warning(
            IssueCode.INVALID_DATE,
            f"Due date equals the invoice date ({invoice.due_date}) despite '{invoice.payment_terms}' terms; "
            "the vendor may be pressing for early payment.",
            field="due_date", actual=str(invoice.due_date),
        )]
    return []


def check_currency(invoice: Invoice) -> list[ValidationIssue]:
    if invoice.currency == "USD":
        return []
    return [_warning(
        IssueCode.UNSUPPORTED_CURRENCY,
        f"Invoice is in {invoice.currency}; approval thresholds are in USD, so the amount needs conversion before review.",
        field="currency", expected="USD", actual=invoice.currency,
    )]


def check_suspicious_content(invoice: Invoice) -> list[ValidationIssue]:
    text = " ".join(filter(None, (invoice.notes, invoice.payment_terms)))
    found = sorted({m.group(0).lower() for m in _SUSPICIOUS.finditer(text)})
    if not found:
        return []
    return [_warning(
        IssueCode.SUSPICIOUS_CONTENT,
        f"Invoice uses pressure language ({', '.join(found)}), a common pattern in payment fraud.",
        field="notes", actual=text,
    )]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def validate_invoice(invoice: Invoice, inventory: InventoryLookup) -> ValidationResult:
    """Run every check and return a ValidationResult.

    Raises:
        InventoryDatabaseError: If inventory cannot be queried.
    """
    checks: list[Callable[[Invoice], list[ValidationIssue]]] = [
        check_required_fields,
        check_extraction_warnings,
        check_quantities,
        make_inventory_check(inventory),
        check_amounts,
        check_dates,
        check_currency,
        check_suspicious_content,
    ]
    issues = [issue for check in checks for issue in check(invoice)]
    result = ValidationResult(invoice_number=invoice.invoice_number, issues=issues)

    logger.info(
        "Validated %s: %s (%d errors, %d warnings)",
        invoice.invoice_number, "valid" if result.is_valid else "INVALID", len(result.errors), len(result.warnings),
    )
    for issue in issues:
        logger.debug("%s [%s/%s] %s", invoice.invoice_number, issue.severity.value, issue.code.value, issue.message)
    return result
