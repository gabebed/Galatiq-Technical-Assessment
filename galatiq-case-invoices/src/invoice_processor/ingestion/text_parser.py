"""Rule-based parser for free-text invoices (TXT files and text extracted from PDFs).

Handles "Label: value" headers (including several on one line, as pdfplumber
produces for INV-1013), abbreviated/misspelled labels (INV-1002), email
wrappers (INV-1008), and three line-item layouts:

    WidgetA    qty: 10    unit price: $250.00      (INV-1001, 1002, 1003)
    - SuperGizmo       x12     $400.00 each          (INV-1008)
    WidgetA   8   $250.00   $2,000.00  [note]        (tables: 1010-1013)
"""

from __future__ import annotations

import re

from invoice_processor.ingestion.builder import RawInvoice, RawItem

# (pattern, field, rank). When several labels map to one field, the lowest rank
# wins; ties keep the first occurrence. Longer variants come first so that
# "Due Date" is not read as "Due" and "Total Amount" is not read as "Total".
_LABELS: tuple[tuple[str, str, int], ...] = (
    (r"invoice\s+(?:number|no\.?|\#)", "invoice_number", 0),
    (r"inv\s*(?:no\.?|\#)", "invoice_number", 0),
    (r"invoice\s+date", "invoice_date", 0),
    (r"invoice", "invoice_number", 1),
    (r"vendor|vndr", "vendor_name", 0),
    (r"from", "vendor_name", 1),  # loses to "Vendor:" in email-wrapped invoices
    (r"due\s+(?:date|dt)", "due_date", 0),
    (r"due", "due_date", 0),
    (r"date|dt", "invoice_date", 0),
    (r"sub\s*total", "subtotal", 0),
    (r"(?:sales\s+)?tax(?:\s*\([^)]*\))?", "tax_amount", 0),
    (r"shipping", "shipping", 0),
    (r"grand\s+total|total\s+amount|amount\s+due|total", "total", 0),
    (r"amt", "total", 1),
    (r"(?:payment|pymnt)\s+terms|terms", "payment_terms", 0),
    (r"notes?", "notes", 0),
)

_LABEL_RE = re.compile(
    r"(?:^|(?<=\s))(?:"
    + "|".join(f"(?P<l{i}>{pattern})" for i, (pattern, _, _) in enumerate(_LABELS))
    + r")\s*:",
    re.IGNORECASE,
)
_TAX_RATE = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*%\s*\)")
_INVOICE_ID = re.compile(r"\bINV[\s#-]*(\d{3,})\b", re.IGNORECASE)

_NAME = r"(?P<name>[A-Za-z][^:$]*?)"
_NUMBER = r"-?[\d,.Oo]*\d[\d,.Oo]*"  # tolerates OCR 'O' for zero; cleaned up by normalize
_PRICE = rf"\$?\s*{_NUMBER}"
_DOLLARS = rf"\$\s*{_NUMBER}"  # tables require '$' so stray numbers aren't read as items
_ITEM_PATTERNS = (
    re.compile(
        rf"^{_NAME}\s+qty\s*:?\s*(?P<quantity>-?\d+)\s+(?:unit\s+price\s*:?|@)\s*(?P<unit_price>{_PRICE})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^[-*•]\s*{_NAME}\s+x\s*(?P<quantity>-?\d+)\s+(?P<unit_price>{_PRICE})(?:\s*(?:each|ea))?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^{_NAME}\s+(?P<quantity>-?\d+)\s+(?P<unit_price>{_DOLLARS})"
        rf"(?:\s+(?P<line_total>{_DOLLARS})(?:\s+(?P<note>[A-Za-z].*?))?)?\s*$"
    ),
)


class LabelAssigner:
    """Assigns labelled values to a RawInvoice, honouring label rank."""

    def __init__(self, raw: RawInvoice) -> None:
        self.raw = raw
        self._ranks: dict[str, int] = {}

    def assign(self, field: str, label: str, value: str, rank: int = 0) -> None:
        if not value or self._ranks.get(field, rank + 1) <= rank:
            return
        self._ranks[field] = rank
        setattr(self.raw, field, value)
        if field == "tax_amount" and (rate := _TAX_RATE.search(label)):
            self.raw.tax_rate = f"{rate.group(1)}%"


def split_labeled(line: str) -> list[tuple[str, str, str, int]]:
    """Split a line beginning with a known label into (field, label, value, rank) tuples."""
    matches = list(_LABEL_RE.finditer(line))
    if not matches or matches[0].start() != 0:
        return []
    pairs = []
    for match, following in zip(matches, [*matches[1:], None]):
        _, field, rank = _LABELS[int(match.lastgroup[1:])]
        end = following.start() if following else len(line)
        label = match.group(match.lastgroup)
        pairs.append((field, label, line[match.end():end].strip(), rank))
    return pairs


def match_label(text: str) -> tuple[str, str] | None:
    """Return (field, label) if ``text`` is exactly a known label such as 'Tax (6%):'."""
    match = _LABEL_RE.fullmatch(text.strip())
    if not match:
        return None
    return _LABELS[int(match.lastgroup[1:])][1], match.group(match.lastgroup)


def _match_item(line: str) -> RawItem | None:
    for pattern in _ITEM_PATTERNS:
        if match := pattern.match(line):
            return RawItem(**match.groupdict())
    return None


def parse_text(text: str) -> RawInvoice:
    raw = RawInvoice()
    assigner = LabelAssigner(raw)
    in_notes = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            in_notes = False
            continue
        if item := _match_item(stripped):
            raw.items.append(item)
            in_notes = False
            continue
        pairs = split_labeled(stripped)
        if pairs:
            for field, label, value, rank in pairs:
                assigner.assign(field, label, value, rank)
            in_notes = pairs[-1][0] == "notes"
        elif in_notes:  # wrapped NOTES block (INV-1012)
            raw.notes = f"{raw.notes} {stripped}"

    if raw.invoice_number is None and (match := _INVOICE_ID.search(text)):
        raw.invoice_number = match.group(1)  # e.g. "INVOICE #INV-1010" header
    return raw
