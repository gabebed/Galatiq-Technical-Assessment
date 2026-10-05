"""Ingestion tests against the real sample invoices in data/invoices/.

Malformed-file tests derive their inputs from those same files (truncated,
re-encoded, renamed) rather than inventing invoice content.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from invoice_processor.ingestion import IngestionError, extract_invoice
from invoice_processor.ingestion.normalize import (
    normalize_invoice_number,
    normalize_item_name,
    parse_date,
    parse_money,
    parse_quantity,
)

INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"

# file -> (invoice_number, vendor, stated total, due date, [(item, qty), ...])
EXPECTED = {
    "invoice_1001.txt": ("INV-1001", "Widgets Inc.", "5000.00", date(2026, 2, 1), [("WidgetA", 10), ("WidgetB", 5)]),
    "invoice_1002.txt": ("INV-1002", "Gadgets Co.", "15000.00", date(2026, 1, 30), [("GadgetX", 20)]),
    "invoice_1003.txt": ("INV-1003", "Fraudster LLC", "100000.00", None, [("FakeItem", 100)]),
    "invoice_1004.json": ("INV-1004", "Precision Parts Ltd.", "1890.00", date(2026, 2, 22),
                          [("WidgetA", 3), ("WidgetB", 2)]),
    "invoice_1004_revised.json": ("INV-1004", "Precision Parts Ltd.", "5940.00", date(2026, 2, 22),
                                  [("WidgetA", 3), ("WidgetB", 2), ("GadgetX", 5)]),
    "invoice_1005.json": ("INV-1005", "Global Supply Chain Partners", "15225.00", date(2026, 3, 18),
                          [("WidgetA", 14), ("GadgetX", 8), ("WidgetB", 10)]),
    "invoice_1006.csv": ("INV-1006", "Acme Industrial Supplies", "2750.00", date(2026, 2, 10),
                         [("WidgetA", 5), ("WidgetB", 3)]),
    "invoice_1007.csv": ("INV-1007", "MegaWidgets Corp", "15525.00", date(2026, 2, 28),
                         [("WidgetA", 20), ("WidgetB", 15), ("GadgetX", 3)]),
    "invoice_1008.txt": ("INV-1008", "NoProd Industries", "9900.00", date(2026, 1, 20),
                         [("SuperGizmo", 12), ("MegaSprocket", 6)]),
    "invoice_1009.json": ("INV-1009", None, "-250.00", None, [("WidgetA", -5), ("WidgetB", 2)]),
    "invoice_1010.txt": ("INV-1010", "Consolidated Materials Group", "7185.00", date(2026, 2, 26),
                         [("WidgetA", 8), ("WidgetB", 4), ("GadgetX", 2), ("WidgetA", 4)]),
    "invoice_1011.txt": ("INV-1011", "Summit Manufacturing Co.", "3000.00", date(2026, 2, 20),
                         [("WidgetA", 6), ("WidgetB", 3)]),
    "invoice_1011.pdf": ("INV-1011", "Summit Manufacturing Co.", "3000.00", date(2026, 2, 20),
                         [("WidgetA", 6), ("WidgetB", 3)]),
    "invoice_1012.txt": ("INV-1012", "QuickShip Distributers", "9975.00", date(2026, 2, 25),
                         [("WidgetA", 12), ("WidgetB", 7), ("GadgetX", 4)]),
    "invoice_1012.pdf": ("INV-1012", "QuickShip Distributers", "9975.00", date(2026, 2, 25),
                         [("WidgetA", 12), ("WidgetB", 7), ("GadgetX", 4)]),
    "invoice_1013.json": ("INV-1013", "Atlas Industrial Supply", "22562.80", date(2026, 3, 24),
                          [("WidgetA", 15), ("WidgetB", 10), ("GadgetX", 5), ("WidgetA", 5),
                           ("WidgetB", 8), ("GadgetX", 3), ("WidgetA", 2), ("GadgetX", 1)]),
    "invoice_1013.pdf": ("INV-1013", "Atlas Industrial Supply", "22562.80", date(2026, 3, 24),
                         [("WidgetA", 15), ("WidgetB", 10), ("GadgetX", 5), ("WidgetA", 5),
                          ("WidgetB", 8), ("GadgetX", 3), ("WidgetA", 2), ("GadgetX", 1)]),
    "invoice_1014.xml": ("INV-1014", "TechParts International", "4125.00", date(2026, 2, 26),
                         [("WidgetA", 4), ("WidgetB", 6)]),
    "invoice_1015.csv": ("INV-1015", "Reliable Components Inc.", "6500.00", date(2026, 2, 28),
                         [("WidgetA", 10), ("WidgetB", 5), ("GadgetX", 2)]),
    "invoice_1016.json": ("INV-1016", "Widgets Inc.", "3233.00", date(2026, 2, 27),
                          [("WidgetA", 4), ("WidgetB", 2), ("WidgetC", 3)]),
}


def test_every_sample_invoice_has_an_expectation() -> None:
    assert {p.name for p in INVOICES.iterdir() if p.is_file()} == set(EXPECTED)


@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_extracts_core_fields(filename: str) -> None:
    number, vendor, total, due, items = EXPECTED[filename]
    invoice = extract_invoice(INVOICES / filename)

    assert invoice.invoice_number == number
    assert invoice.vendor_name == vendor
    assert invoice.total == Decimal(total)
    assert invoice.due_date == due
    assert [(i.name, i.quantity) for i in invoice.items] == items
    assert invoice.source_file == str(INVOICES / filename)


@pytest.mark.parametrize(
    "filename",
    ["invoice_1001.txt", "invoice_1004.json", "invoice_1006.csv", "invoice_1011.txt", "invoice_1015.csv",
     "invoice_1013.pdf", "invoice_1014.xml"],
)
def test_clean_invoices_produce_no_warnings(filename: str) -> None:
    assert extract_invoice(INVOICES / filename).extraction_warnings == []


# --------------------------------------------------------------------------- #
# Per-invoice edge cases from the analysis matrix
# --------------------------------------------------------------------------- #


def test_1002_abbreviated_labels() -> None:
    invoice = extract_invoice(INVOICES / "invoice_1002.txt")
    assert invoice.invoice_date == date(2026, 1, 30)
    assert invoice.payment_terms == "Net 30"
    assert invoice.items[0].unit_price == Decimal("750")


def test_1003_unparseable_due_date_is_none_with_warning() -> None:
    invoice = extract_invoice(INVOICES / "invoice_1003.txt")
    assert invoice.due_date is None
    assert any("yesterday" in w for w in invoice.extraction_warnings)
    assert "Wire transfer preferred" in invoice.notes


def test_1004_revision_is_captured() -> None:
    assert extract_invoice(INVOICES / "invoice_1004.json").revision is None
    assert extract_invoice(INVOICES / "invoice_1004_revised.json").revision == "R1"


def test_1007_us_dates_and_summary_rows() -> None:
    invoice = extract_invoice(INVOICES / "invoice_1007.csv")
    assert invoice.invoice_date == date(2026, 1, 28)
    assert (invoice.subtotal, invoice.tax_amount, invoice.tax_rate) == (
        Decimal("14750.00"), Decimal("885.00"), Decimal("0.06"))
    # Stated total is preserved as-is; the $110 discrepancy is validation's job.
    assert invoice.computed_subtotal + invoice.tax_amount == Decimal("15635.00")


def test_1008_vendor_comes_from_body_not_email_header() -> None:
    assert extract_invoice(INVOICES / "invoice_1008.txt").vendor_name == "NoProd Industries"


def test_1009_missing_and_invalid_values_are_preserved() -> None:
    invoice = extract_invoice(INVOICES / "invoice_1009.json")
    assert invoice.vendor_address is None
    assert invoice.payment_terms is None
    assert invoice.subtotal == Decimal("1000.00")


def test_1010_qualifier_shipping_and_long_dates() -> None:
    invoice = extract_invoice(INVOICES / "invoice_1010.txt")
    rush = invoice.items[3]
    assert (rush.name, rush.unit_price, rush.note) == ("WidgetA", Decimal("300.00"), "rush order")
    assert invoice.shipping == Decimal("150.00")
    assert invoice.invoice_date == date(2026, 1, 27)
    assert invoice.item_quantities()["WidgetA"] == 12


def test_1011_pdf_lacks_subtotal_present_in_txt() -> None:
    assert extract_invoice(INVOICES / "invoice_1011.txt").subtotal == Decimal("3000.00")
    assert extract_invoice(INVOICES / "invoice_1011.pdf").subtotal is None


@pytest.mark.parametrize("filename", ["invoice_1012.txt", "invoice_1012.pdf"])
def test_1012_ocr_artifacts_are_corrected_and_reported(filename: str) -> None:
    invoice = extract_invoice(INVOICES / filename)
    assert invoice.invoice_date == date(2026, 1, 26)
    assert invoice.items[1].line_total == Decimal("3500.00")
    assert invoice.tax_rate == Decimal("0.05")
    assert invoice.notes.startswith("Ref PO-20260115") and invoice.notes.endswith("questions.")
    assert len(invoice.extraction_warnings) == 2
    assert all("OCR" in w for w in invoice.extraction_warnings)


def test_1013_pdf_matches_json() -> None:
    pdf = extract_invoice(INVOICES / "invoice_1013.pdf")
    js = extract_invoice(INVOICES / "invoice_1013.json")
    assert pdf.items == js.items
    assert (pdf.subtotal, pdf.tax_amount, pdf.total) == (js.subtotal, js.tax_amount, js.total)
    assert pdf.item_quantities() == {"WidgetA": 22, "WidgetB": 18, "GadgetX": 9}


def test_1014_xml_currency() -> None:
    invoice = extract_invoice(INVOICES / "invoice_1014.xml")
    assert invoice.currency == "EUR"
    assert invoice.tax_rate == Decimal("0.10")


# --------------------------------------------------------------------------- #
# Malformed and unsupported files
# --------------------------------------------------------------------------- #


def _derived(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="Could not read"):
        extract_invoice(tmp_path / "invoice_9999.txt")


def test_unsupported_extension(tmp_path: Path) -> None:
    path = _derived(tmp_path, "invoice_1001.docx", (INVOICES / "invoice_1001.txt").read_bytes())
    with pytest.raises(IngestionError, match="Unsupported file type"):
        extract_invoice(path)


@pytest.mark.parametrize(
    ("source", "name", "match"),
    [
        ("invoice_1004.json", "truncated.json", "Malformed JSON"),
        ("invoice_1014.xml", "truncated.xml", "Malformed XML"),
        ("invoice_1011.pdf", "truncated.pdf", "Could not read PDF"),
    ],
)
def test_truncated_files(tmp_path: Path, source: str, name: str, match: str) -> None:
    data = (INVOICES / source).read_bytes()
    with pytest.raises(IngestionError, match=match):
        extract_invoice(_derived(tmp_path, name, data[: len(data) // 2]))


def test_json_that_is_not_an_object(tmp_path: Path) -> None:
    data = b"[" + (INVOICES / "invoice_1004.json").read_bytes() + b"]"
    with pytest.raises(IngestionError, match="JSON object"):
        extract_invoice(_derived(tmp_path, "wrapped.json", data))


def test_csv_without_recognizable_header(tmp_path: Path) -> None:
    lines = (INVOICES / "invoice_1007.csv").read_bytes().splitlines(keepends=True)
    with pytest.raises(IngestionError, match="Unrecognized CSV layout"):
        extract_invoice(_derived(tmp_path, "headerless.csv", b"".join(lines[1:])))


def test_non_utf8_text(tmp_path: Path) -> None:
    data = (INVOICES / "invoice_1001.txt").read_text(encoding="utf-8").encode("utf-16")
    with pytest.raises(IngestionError, match="UTF-8"):
        extract_invoice(_derived(tmp_path, "utf16.txt", data))


@pytest.mark.parametrize("name", ["empty.txt", "empty.csv", "empty.json"])
def test_empty_files(tmp_path: Path, name: str) -> None:
    with pytest.raises(IngestionError):
        extract_invoice(_derived(tmp_path, name, b""))


def test_text_without_invoice_data(tmp_path: Path) -> None:
    # The sign-off lines of the INV-1008 email contain no invoice fields.
    tail = (INVOICES / "invoice_1008.txt").read_bytes().splitlines(keepends=True)[-4:]
    with pytest.raises(IngestionError, match="No invoice data"):
        extract_invoice(_derived(tmp_path, "signoff.txt", b"".join(tail)))


def test_missing_and_bad_fields_degrade_gracefully(tmp_path: Path) -> None:
    text = (INVOICES / "invoice_1004.json").read_text(encoding="utf-8")
    text = text.replace('"due_date": "2026-02-22"', '"due_date": null')
    text = text.replace('"quantity": 3', '"quantity": "three"')
    invoice = extract_invoice(_derived(tmp_path, "degraded.json", text.encode()))

    assert invoice.due_date is None
    assert invoice.items[0].quantity is None
    assert invoice.vendor_name == "Precision Parts Ltd."
    assert any("three" in w for w in invoice.extraction_warnings)


# --------------------------------------------------------------------------- #
# Normalizers (inputs taken from the sample files)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-02-01", date(2026, 2, 1)),
        ("Jan 30 2026", date(2026, 1, 30)),
        ("01/28/2026", date(2026, 1, 28)),
        ("January 27, 2026", date(2026, 1, 27)),
        ("25-Feb-2026", date(2026, 2, 25)),
        ("26-Jan-2O26", date(2026, 1, 26)),
        ("yesterday", None),
        (None, None),
    ],
)
def test_parse_date(raw: str | None, expected: date | None) -> None:
    assert parse_date(raw, "date", []) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("$5,000.00", "5000.00"), ("$750", "750"), ("$3,500.O0", "3500.00"), (1472.80, "1472.8"),
     (-250.00, "-250.0"), ("Net 30", None)],
)
def test_parse_money(raw: object, expected: str | None) -> None:
    result = parse_money(raw, "amount", [])
    assert result == (Decimal(expected) if expected else None)


def test_parse_quantity_rejects_fractions() -> None:
    warnings: list[str] = []
    assert parse_quantity(2.5, "qty", warnings) is None and warnings
    assert parse_quantity("-5", "qty", []) == -5


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("INV-1001", "INV-1001"), ("1002", "INV-1002"), ("INV 1012", "INV-1012"), (None, None)],
)
def test_normalize_invoice_number(raw: str | None, expected: str | None) -> None:
    assert normalize_invoice_number(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("WidgetA", ("WidgetA", None)), ("Widget A", ("WidgetA", None)), ("Gadget X", ("GadgetX", None)),
     ("WidgetA (rush order)", ("WidgetA", "rush order")), ("SuperGizmo", ("SuperGizmo", None)), ("  ", (None, None))],
)
def test_normalize_item_name(raw: str, expected: tuple) -> None:
    assert normalize_item_name(raw) == expected
