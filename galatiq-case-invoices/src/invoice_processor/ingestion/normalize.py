"""Deterministic normalizers from raw document values to typed values.

Normalizers never raise on bad input. They return ``None`` and append a
human-readable note to ``warnings`` so the validation stage can distinguish
"missing" from "present but unreadable".
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

# Letter O misread for zero, only when adjacent to digits/separators ("2O26", "3,500.O0").
_OCR_ZERO = re.compile(r"(?<=[\d,.])[Oo]|[Oo](?=[\d,.])")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_INVOICE_NUMBER = re.compile(r"^(?:INV(?:OICE)?)?[\s#:-]*(\d+)$", re.IGNORECASE)
_SPLIT_SUFFIX = re.compile(r"^([A-Za-z]+) ([A-Z0-9])$")  # "Widget A" -> "WidgetA"
_PARENTHETICAL = re.compile(r"^(.*?)\s*\(([^)]*)\)\s*$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")

# Slash dates are read as US month/day: Acme Corp is a US company.
DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%b %d %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%B %d, %Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d %b %Y",
    "%d %B %Y",
)


def clean_text(value: Any) -> str | None:
    """Collapse whitespace; empty values become None."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _fix_ocr_digits(text: str, field: str, warnings: list[str]) -> str:
    fixed = _OCR_ZERO.sub("0", text)
    if fixed != text:
        warnings.append(f"{field}: corrected OCR artifact {text!r} -> {fixed!r}")
    return fixed


def parse_money(value: Any, field: str, warnings: list[str]) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        warnings.append(f"{field}: expected an amount, got {value!r}")
        return None
    if isinstance(value, (int, Decimal)):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))

    text = clean_text(value)
    if text is None:
        return None
    cleaned = re.sub(r"[\s$€£,]|USD|EUR", "", _fix_ocr_digits(text, field, warnings), flags=re.IGNORECASE)
    if not _NUMBER.fullmatch(cleaned):
        warnings.append(f"{field}: could not parse {text!r} as an amount")
        return None
    return Decimal(cleaned)


def parse_quantity(value: Any, field: str, warnings: list[str]) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        warnings.append(f"{field}: expected a quantity, got {value!r}")
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        warnings.append(f"{field}: quantity {value!r} is not a whole number")
        return None

    text = clean_text(value)
    if text is None:
        return None
    cleaned = _fix_ocr_digits(text, field, warnings).replace(",", "")
    if not re.fullmatch(r"-?\d+", cleaned):
        warnings.append(f"{field}: could not parse {text!r} as a quantity")
        return None
    return int(cleaned)


def parse_rate(value: Any, field: str, warnings: list[str]) -> Decimal | None:
    """Parse a tax rate as a fraction. '5%' -> 0.05; numeric values are taken as fractions."""
    if value is None or isinstance(value, bool):
        return None
    text = clean_text(value)
    if text is None:
        return None
    is_percent = text.endswith("%")
    try:
        rate = Decimal(text.rstrip("%").strip())
    except InvalidOperation:
        warnings.append(f"{field}: could not parse {text!r} as a rate")
        return None
    return rate / 100 if is_percent else rate


def parse_date(value: Any, field: str, warnings: list[str]) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = clean_text(value)
    if text is None:
        return None
    candidate = _fix_ocr_digits(text, field, warnings).rstrip(".")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    warnings.append(f"{field}: could not parse {text!r} as a date")
    return None


def parse_currency(value: Any) -> str | None:
    """Return an ISO 4217 code, or None if absent. Raises ValueError if malformed."""
    text = clean_text(value)
    if text is None:
        return None
    code = text.upper()
    if not _CURRENCY.fullmatch(code):
        raise ValueError(f"unrecognized currency {text!r}")
    return code


def normalize_invoice_number(value: Any) -> str | None:
    """'INV-1001', 'INV 1012', '#1002', '1002' -> 'INV-<digits>'. Other formats are kept as-is."""
    text = clean_text(value)
    if text is None:
        return None
    match = _INVOICE_NUMBER.fullmatch(text)
    return f"INV-{match.group(1)}" if match else text


def normalize_item_name(value: Any) -> tuple[str | None, str | None]:
    """Split an item label into (inventory name, qualifier).

    'WidgetA (rush order)' -> ('WidgetA', 'rush order'); 'Widget A' -> ('WidgetA', None).
    """
    text = clean_text(value)
    if text is None:
        return None, None
    note = None
    match = _PARENTHETICAL.fullmatch(text)
    if match:
        text, note = match.group(1), clean_text(match.group(2))
    text = _SPLIT_SUFFIX.sub(r"\1\2", text)
    return (text or None), note
