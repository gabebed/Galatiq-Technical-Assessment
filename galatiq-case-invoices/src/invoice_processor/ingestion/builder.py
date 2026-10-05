"""Raw, format-agnostic extraction results and their conversion to ``Invoice``.

Format parsers only locate values; all type conversion and cleanup happens
here so every format is normalized identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from invoice_processor.ingestion.errors import IngestionError
from invoice_processor.ingestion.normalize import (
    clean_text,
    normalize_invoice_number,
    normalize_item_name,
    parse_currency,
    parse_date,
    parse_money,
    parse_quantity,
    parse_rate,
)
from invoice_processor.models import Invoice, InvoiceItem


@dataclass
class RawItem:
    name: Any = None
    quantity: Any = None
    unit_price: Any = None
    line_total: Any = None
    note: Any = None


@dataclass
class RawInvoice:
    invoice_number: Any = None
    revision: Any = None
    vendor_name: Any = None
    vendor_address: Any = None
    invoice_date: Any = None
    due_date: Any = None
    currency: Any = None
    subtotal: Any = None
    tax_rate: Any = None
    tax_amount: Any = None
    shipping: Any = None
    total: Any = None
    payment_terms: Any = None
    notes: Any = None
    items: list[RawItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_invoice(raw: RawInvoice, *, source_file: str | None = None) -> Invoice:
    """Normalize raw values into an ``Invoice``.

    Raises:
        IngestionError: If nothing invoice-like was found or a value is
            malformed beyond recovery (e.g. an unrecognized currency).
    """
    warnings = list(raw.warnings)
    items = [item for i, r in enumerate(raw.items) if (item := _build_item(r, i, warnings))]

    try:
        currency = parse_currency(raw.currency)
    except ValueError as exc:
        raise IngestionError(str(exc)) from exc

    fields: dict[str, Any] = {
        "invoice_number": normalize_invoice_number(raw.invoice_number),
        "revision": clean_text(raw.revision),
        "vendor_name": clean_text(raw.vendor_name),
        "vendor_address": clean_text(raw.vendor_address),
        "invoice_date": parse_date(raw.invoice_date, "invoice_date", warnings),
        "due_date": parse_date(raw.due_date, "due_date", warnings),
        "items": items,
        "subtotal": parse_money(raw.subtotal, "subtotal", warnings),
        "tax_rate": parse_rate(raw.tax_rate, "tax_rate", warnings),
        "tax_amount": parse_money(raw.tax_amount, "tax_amount", warnings),
        "shipping": parse_money(raw.shipping, "shipping", warnings),
        "total": parse_money(raw.total, "total", warnings),
        "payment_terms": clean_text(raw.payment_terms),
        "notes": clean_text(raw.notes),
        "source_file": source_file,
        "extraction_warnings": warnings,
    }
    if currency:
        fields["currency"] = currency

    if not any((fields["invoice_number"], fields["vendor_name"], items, fields["total"])):
        raise IngestionError("No invoice data found (no invoice number, vendor, items, or total)")

    try:
        return Invoice(**fields)
    except ValidationError as exc:  # normalizers should prevent this; fail loudly if not
        raise IngestionError(f"Extracted data does not fit the Invoice model: {exc}") from exc


def _build_item(raw: RawItem, index: int, warnings: list[str]) -> InvoiceItem | None:
    prefix = f"items[{index}]"
    name, qualifier = normalize_item_name(raw.name)
    if name is None:
        warnings.append(f"{prefix}: skipped line with no item name")
        return None
    notes = [n for n in (qualifier, clean_text(raw.note)) if n]
    return InvoiceItem(
        name=name,
        quantity=parse_quantity(raw.quantity, f"{prefix}.quantity", warnings),
        unit_price=parse_money(raw.unit_price, f"{prefix}.unit_price", warnings),
        line_total=parse_money(raw.line_total, f"{prefix}.line_total", warnings),
        note="; ".join(notes) or None,
    )
