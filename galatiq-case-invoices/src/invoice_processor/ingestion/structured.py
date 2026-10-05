"""Parsers for structured formats: JSON, CSV (two layouts), and XML."""

from __future__ import annotations

import csv
import io
import json
import xml.etree.ElementTree as ET
from typing import Any

from invoice_processor.ingestion.builder import RawInvoice, RawItem
from invoice_processor.ingestion.errors import IngestionError
from invoice_processor.ingestion.text_parser import LabelAssigner, match_label

# Header-level keys shared by JSON and key/value CSV, mapped to RawInvoice fields.
_HEADER_KEYS = {
    "invoice_number": "invoice_number",
    "invoice_id": "invoice_number",
    "revision": "revision",
    "vendor": "vendor_name",
    "vendor_name": "vendor_name",
    "date": "invoice_date",
    "invoice_date": "invoice_date",
    "due_date": "due_date",
    "currency": "currency",
    "subtotal": "subtotal",
    "tax_rate": "tax_rate",
    "tax": "tax_amount",
    "tax_amount": "tax_amount",
    "shipping": "shipping",
    "total": "total",
    "payment_terms": "payment_terms",
    "terms": "payment_terms",
    "notes": "notes",
}
_ITEM_KEYS = {
    "item": "name",
    "name": "name",
    "description": "name",
    "quantity": "quantity",
    "qty": "quantity",
    "unit_price": "unit_price",
    "price": "unit_price",
    "amount": "line_total",
    "line_total": "line_total",
    "note": "note",
    "notes": "note",
}


def _key(text: str) -> str:
    return "_".join(text.strip().lower().split())


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


def parse_json(text: str) -> RawInvoice:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IngestionError(f"Malformed JSON: {exc}") from exc
    except RecursionError as exc:  # pathologically nested input
        raise IngestionError("Malformed JSON: nesting is too deep") from exc
    if not isinstance(data, dict):
        raise IngestionError(f"Expected a JSON object at top level, got {type(data).__name__}")

    raw = RawInvoice()
    for key, value in data.items():
        if (field := _HEADER_KEYS.get(_key(key))) and field != "vendor_name":
            setattr(raw, field, value)

    vendor = data.get("vendor", data.get("vendor_name"))
    if isinstance(vendor, dict):
        raw.vendor_name, raw.vendor_address = vendor.get("name"), vendor.get("address")
    else:
        raw.vendor_name = vendor

    lines = data.get("line_items", data.get("items"))
    if lines is not None and not isinstance(lines, list):
        raw.warnings.append("line_items: expected a list; ignored")
        lines = None
    for index, line in enumerate(lines or []):
        if not isinstance(line, dict):
            raw.warnings.append(f"items[{index}]: expected an object; skipped")
            continue
        item = RawItem()
        for key, value in line.items():
            if attr := _ITEM_KEYS.get(_key(key)):
                setattr(item, attr, value)
        raw.items.append(item)
    return raw


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def parse_csv(text: str) -> RawInvoice:
    """Parse either a two-column ``field,value`` CSV or a one-row-per-item table."""
    try:
        rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text))]
    except csv.Error as exc:
        raise IngestionError(f"Malformed CSV: {exc}") from exc
    rows = [row for row in rows if any(row)]
    if not rows:
        raise IngestionError("CSV file is empty")

    header = [_key(cell) for cell in rows[0]]
    if header[:2] == ["field", "value"]:
        return _parse_key_value_csv(rows[1:])
    if "item" in header and ({"qty", "quantity"} & set(header)):
        return _parse_tabular_csv(header, rows[1:])
    raise IngestionError(f"Unrecognized CSV layout (header: {rows[0]})")


def _parse_key_value_csv(rows: list[list[str]]) -> RawInvoice:
    """Rows of ``key,value``; each ``item`` row starts a new line item (INV-1006)."""
    raw = RawInvoice()
    current: RawItem | None = None
    for row_number, row in enumerate(rows, start=2):
        if len(row) < 2:
            raw.warnings.append(f"row {row_number}: expected 'field,value'; skipped")
            continue
        key, value = _key(row[0]), row[1]
        if key in ("item", "name"):
            current = RawItem(name=value)
            raw.items.append(current)
        elif key in _ITEM_KEYS:
            if current is None:
                raw.warnings.append(f"row {row_number}: {key!r} appears before any item; skipped")
            else:
                setattr(current, _ITEM_KEYS[key], value)
        elif field := _HEADER_KEYS.get(key):
            setattr(raw, field, value)
    return raw


_TABLE_COLUMNS = {
    "invoice_number": "invoice_number",
    "vendor": "vendor_name",
    "date": "invoice_date",
    "due_date": "due_date",
    "currency": "currency",
}


def _parse_tabular_csv(header: list[str], rows: list[list[str]]) -> RawInvoice:
    """One row per item with repeated header columns, then label/value summary rows (INV-1007)."""
    raw = RawInvoice()
    assigner = LabelAssigner(raw)
    header_values: dict[str, list[str]] = {}

    for row in rows:
        cells = dict(zip(header, row))
        if cells.get("item"):
            raw.items.append(
                RawItem(**{_ITEM_KEYS[col]: val for col, val in cells.items() if col in _ITEM_KEYS and val})
            )
            for col, field in _TABLE_COLUMNS.items():
                if cells.get(col):
                    header_values.setdefault(field, []).append(cells[col])
            continue
        # Summary row such as ",,,,,,Subtotal:,14750.00"
        non_empty = [cell for cell in row if cell]
        for label_cell, value in zip(non_empty, non_empty[1:]):
            if labeled := match_label(label_cell):
                assigner.assign(labeled[0], labeled[1], value)

    for field, values in header_values.items():
        setattr(raw, field, values[0])
        if len(set(values)) > 1:
            raw.warnings.append(f"{field}: rows disagree ({sorted(set(values))}); using {values[0]!r}")
    return raw


# --------------------------------------------------------------------------- #
# XML
# --------------------------------------------------------------------------- #


def parse_xml(data: bytes) -> RawInvoice:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise IngestionError(f"Malformed XML: {exc}") from exc

    def text(element: ET.Element, *paths: str) -> Any:
        for path in paths:
            if (value := element.findtext(path)) is not None:
                return value
        return None

    raw = RawInvoice(
        invoice_number=text(root, ".//invoice_number"),
        revision=text(root, ".//revision"),
        invoice_date=text(root, ".//header/date", ".//invoice_date", ".//date"),
        due_date=text(root, ".//due_date"),
        currency=text(root, ".//currency"),
        subtotal=text(root, ".//subtotal"),
        tax_rate=text(root, ".//tax_rate"),
        tax_amount=text(root, ".//tax_amount"),
        shipping=text(root, ".//shipping"),
        total=text(root, ".//totals/total", ".//total"),
        payment_terms=text(root, ".//payment_terms"),
        notes=text(root, "./notes"),
    )
    vendor = root.find(".//vendor")
    if vendor is not None:
        raw.vendor_name = text(vendor, "name") if len(vendor) else vendor.text
        raw.vendor_address = text(vendor, "address")

    for element in root.iterfind(".//line_items/item"):
        raw.items.append(
            RawItem(
                name=text(element, "name", "description"),
                quantity=text(element, "quantity", "qty"),
                unit_price=text(element, "unit_price", "price"),
                line_total=text(element, "amount", "line_total"),
                note=text(element, "note"),
            )
        )
    return raw
