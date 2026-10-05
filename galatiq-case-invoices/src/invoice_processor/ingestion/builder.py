"""Raw, format-agnostic extraction results and their conversion to ``Invoice``.

Format parsers only locate values; all type conversion and cleanup happens
here so every format is normalized identically.
"""

from __future__ import annotations

import re
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
        declared = parse_currency(raw.currency)
    except ValueError as exc:
        raise IngestionError(str(exc)) from exc
    currency = _resolve_currency(declared, _currency_markers(raw))

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


_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
_CURRENCY_CODES = re.compile(r"\b(USD|EUR|GBP)\b", re.IGNORECASE)


def _currency_markers(raw: RawInvoice) -> set[str]:
    """Currencies indicated by symbols/codes on the raw amount strings ('$' is read as USD)."""
    amounts = [raw.subtotal, raw.tax_amount, raw.shipping, raw.total]
    amounts += [value for item in raw.items for value in (item.unit_price, item.line_total)]
    found = set()
    for value in amounts:
        if isinstance(value, str):
            found.update(code for symbol, code in _CURRENCY_SYMBOLS.items() if symbol in value)
            found.update(match.upper() for match in _CURRENCY_CODES.findall(value))
    return found


def _resolve_currency(declared: str | None, markers: set[str]) -> str | None:
    """Never silently treat marked amounts as another currency; ambiguity is a hard error."""
    if len(markers) > 1:
        raise IngestionError(f"Amounts are in more than one currency ({', '.join(sorted(markers))}); "
                             "the amount to pay is ambiguous.")
    if declared and markers and markers != {declared}:
        raise IngestionError(f"Invoice currency is {declared} but amounts are marked {markers.pop()}.")
    return declared or (next(iter(markers)) if markers else None)


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
