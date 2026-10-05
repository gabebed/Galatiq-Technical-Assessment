"""Validation tests: README scenarios and the full sample set as integration tests
(real files + real SQLite DB), plus unit tests per check using an in-memory inventory."""

from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path

import pytest

from invoice_processor.database import InventoryDatabaseError, init_db
from invoice_processor.ingestion import extract_invoice
from invoice_processor.inventory import SQLiteInventory
from invoice_processor.models import Invoice, InvoiceItem, IssueCode, Severity, ValidationResult
from invoice_processor.validation import validate_invoice

INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"
E, W = Severity.ERROR, Severity.WARNING


@pytest.fixture(scope="module")
def inventory(tmp_path_factory: pytest.TempPathFactory) -> SQLiteInventory:
    return SQLiteInventory(init_db(tmp_path_factory.mktemp("db") / "inventory.db"))


def _validate(filename: str, inventory: SQLiteInventory) -> ValidationResult:
    return validate_invoice(extract_invoice(INVOICES / filename), inventory)


def _codes(result: ValidationResult, severity: Severity) -> set[tuple[IssueCode, str | None]]:
    return {(i.code, i.item) for i in result.issues if i.severity is severity}


# --------------------------------------------------------------------------- #
# README scenarios
# --------------------------------------------------------------------------- #


def test_readme_inv_1001_normal_order_passes(inventory: SQLiteInventory) -> None:
    result = _validate("invoice_1001.txt", inventory)
    assert result.is_valid
    assert result.issues == []


def test_readme_inv_1002_quantity_exceeds_stock(inventory: SQLiteInventory) -> None:
    result = _validate("invoice_1002.txt", inventory)
    assert not result.is_valid
    [error] = result.errors
    assert (error.code, error.item, error.expected, error.actual) == (IssueCode.INSUFFICIENT_STOCK, "GadgetX", "<= 5", "20")
    assert "20 units of GadgetX" in error.message and "only 5" in error.message


def test_readme_inv_1003_zero_stock_and_fraud_signals(inventory: SQLiteInventory) -> None:
    result = _validate("invoice_1003.txt", inventory)
    assert not result.is_valid
    assert _codes(result, E) == {(IssueCode.OUT_OF_STOCK, "FakeItem")}
    assert {i.code for i in result.warnings} == {IssueCode.SUSPICIOUS_CONTENT, IssueCode.EXTRACTION_UNCERTAIN}
    assert any("yesterday" in i.message for i in result.warnings)


def test_readme_inv_1008_unknown_items(inventory: SQLiteInventory) -> None:
    result = _validate("invoice_1008.txt", inventory)
    assert not result.is_valid
    assert _codes(result, E) == {(IssueCode.UNKNOWN_ITEM, "SuperGizmo"), (IssueCode.UNKNOWN_ITEM, "MegaSprocket")}


def test_readme_inv_1009_data_integrity(inventory: SQLiteInventory) -> None:
    result = _validate("invoice_1009.json", inventory)
    assert not result.is_valid
    by_field = {(i.code, i.field) for i in result.errors}
    assert (IssueCode.INVALID_QUANTITY, "items[0].quantity") in by_field
    assert (IssueCode.MISSING_FIELD, "vendor_name") in by_field
    assert (IssueCode.INVALID_AMOUNT, "total") in by_field
    assert (IssueCode.AMOUNT_MISMATCH, "subtotal") in by_field
    # The negative line is excluded from stock checks; WidgetB x2 is within stock.
    assert not any(i.code is IssueCode.INSUFFICIENT_STOCK for i in result.issues)


def test_readme_inv_1016_unknown_item(inventory: SQLiteInventory) -> None:
    result = _validate("invoice_1016.json", inventory)
    assert not result.is_valid
    assert _codes(result, E) == {(IssueCode.UNKNOWN_ITEM, "WidgetC")}


# --------------------------------------------------------------------------- #
# Every sample file (matches the edge-case matrix)
# --------------------------------------------------------------------------- #

STOCK = IssueCode.INSUFFICIENT_STOCK
MISMATCH = IssueCode.AMOUNT_MISMATCH

# file -> (is_valid, {(code, item)} of ERRORs, {code} of WARNINGs)
EXPECTED: dict[str, tuple[bool, set, set]] = {
    "invoice_1001.txt": (True, set(), set()),
    "invoice_1002.txt": (False, {(STOCK, "GadgetX")}, {IssueCode.INVALID_DATE}),
    "invoice_1003.txt": (False, {(IssueCode.OUT_OF_STOCK, "FakeItem")},
                         {IssueCode.SUSPICIOUS_CONTENT, IssueCode.EXTRACTION_UNCERTAIN}),
    "invoice_1004.json": (True, set(), set()),
    "invoice_1004_revised.json": (True, set(), set()),  # GadgetX 5 of 5: boundary passes
    "invoice_1005.json": (False, {(STOCK, "GadgetX")}, set()),
    "invoice_1006.csv": (True, set(), set()),
    "invoice_1007.csv": (False, {(STOCK, "WidgetA"), (STOCK, "WidgetB"), (MISMATCH, None)}, set()),
    "invoice_1008.txt": (False, {(IssueCode.UNKNOWN_ITEM, "SuperGizmo"), (IssueCode.UNKNOWN_ITEM, "MegaSprocket")}, set()),
    "invoice_1009.json": (False, {(IssueCode.MISSING_FIELD, None), (IssueCode.INVALID_QUANTITY, "WidgetA"),
                                  (IssueCode.INVALID_AMOUNT, None), (MISMATCH, None)}, {IssueCode.MISSING_FIELD}),
    "invoice_1010.txt": (True, set(), set()),
    "invoice_1011.txt": (True, set(), set()),
    "invoice_1011.pdf": (True, set(), set()),
    "invoice_1012.txt": (True, set(), {IssueCode.EXTRACTION_UNCERTAIN}),
    "invoice_1012.pdf": (True, set(), {IssueCode.EXTRACTION_UNCERTAIN}),
    "invoice_1013.json": (False, {(STOCK, "WidgetA"), (STOCK, "WidgetB"), (STOCK, "GadgetX"), (MISMATCH, None)}, set()),
    "invoice_1013.pdf": (False, {(STOCK, "WidgetA"), (STOCK, "WidgetB"), (STOCK, "GadgetX"), (MISMATCH, None)}, set()),
    "invoice_1014.xml": (True, set(), {IssueCode.UNSUPPORTED_CURRENCY}),
    "invoice_1015.csv": (True, set(), set()),
    "invoice_1016.json": (False, {(IssueCode.UNKNOWN_ITEM, "WidgetC")}, set()),
}


def test_every_sample_invoice_has_an_expectation() -> None:
    assert {p.name for p in INVOICES.iterdir() if p.is_file()} == set(EXPECTED)


@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_sample_invoice_validation(filename: str, inventory: SQLiteInventory) -> None:
    is_valid, errors, warnings = EXPECTED[filename]
    result = _validate(filename, inventory)

    assert result.is_valid is is_valid
    assert _codes(result, E) == errors
    assert {i.code for i in result.warnings} == warnings
    assert result.invoice_number == extract_invoice(INVOICES / filename).invoice_number
    # Hard failures can never coexist with a valid result.
    assert result.is_valid == (not result.errors)
    for issue in result.issues:
        assert len(issue.message) > 20


def test_1013_stock_is_checked_on_combined_quantities(inventory: SQLiteInventory) -> None:
    result = _validate("invoice_1013.json", inventory)
    widget_a = next(i for i in result.errors if i.item == "WidgetA")
    assert (widget_a.actual, widget_a.expected) == ("22", "<= 15")
    assert "across 3 lines" in widget_a.message


def test_1007_total_mismatch_explains_the_difference(inventory: SQLiteInventory) -> None:
    result = _validate("invoice_1007.csv", inventory)
    mismatch = next(i for i in result.errors if i.code is MISMATCH)
    assert (mismatch.field, mismatch.expected, mismatch.actual) == ("total", "15635.00", "15525.00")
    assert "-110.00" in mismatch.message


def test_database_failure_is_not_reported_as_valid(tmp_path: Path) -> None:
    invoice = extract_invoice(INVOICES / "invoice_1001.txt")
    with pytest.raises(InventoryDatabaseError):
        validate_invoice(invoice, SQLiteInventory(tmp_path / "missing.db"))


# --------------------------------------------------------------------------- #
# Unit tests with an in-memory inventory
# --------------------------------------------------------------------------- #


class FakeInventory:
    def __init__(self, stock: dict[str, int]) -> None:
        self.stock = stock
        self.calls: list[list[str]] = []

    def get_stock_levels(self, items: Iterable[str]) -> dict[str, int]:
        names = list(items)
        self.calls.append(names)
        return {n: self.stock[n] for n in names if n in self.stock}


SEED = {"WidgetA": 15, "WidgetB": 10, "GadgetX": 5, "FakeItem": 0}


def _invoice(*items: InvoiceItem, **fields) -> Invoice:
    defaults = {"invoice_number": "INV-1", "vendor_name": "Vendor", "invoice_date": "2026-01-01",
                "due_date": "2026-01-31", "total": sum((i.computed_total or 0 for i in items), Decimal(0))}
    return Invoice(items=list(items), **(defaults | fields))


def _item(name: str, qty: int | None, price: str = "10", **kw) -> InvoiceItem:
    return InvoiceItem(name=name, quantity=qty, unit_price=Decimal(price), **kw)


def test_inventory_is_queried_once_per_invoice() -> None:
    fake = FakeInventory(SEED)
    validate_invoice(_invoice(_item("WidgetA", 1), _item("WidgetB", 1), _item("WidgetA", 1)), fake)
    assert fake.calls == [["WidgetA", "WidgetB"]]


def test_quantity_equal_to_stock_passes() -> None:
    assert validate_invoice(_invoice(_item("GadgetX", 5)), FakeInventory(SEED)).is_valid


@pytest.mark.parametrize("qty", [0, -1, None])
def test_invalid_quantities_are_hard_errors(qty: int | None) -> None:
    result = validate_invoice(_invoice(_item("WidgetA", qty), total=Decimal("10")), FakeInventory(SEED))
    assert not result.is_valid
    assert any(i.code is IssueCode.INVALID_QUANTITY for i in result.errors)


def test_line_total_mismatch() -> None:
    invoice = _invoice(_item("WidgetA", 2, "250", line_total=Decimal("600")), total=Decimal("600"))
    result = validate_invoice(invoice, FakeInventory(SEED))
    assert any(i.code is MISMATCH and i.field == "items[0].line_total" for i in result.errors)


def test_negative_unit_price() -> None:
    result = validate_invoice(_invoice(_item("WidgetA", 1, "-10"), total=Decimal("1")), FakeInventory(SEED))
    assert any(i.code is IssueCode.INVALID_AMOUNT and i.item == "WidgetA" for i in result.errors)


def test_tax_rate_mismatch_is_a_warning() -> None:
    invoice = _invoice(_item("WidgetA", 1, "100"), tax_rate=Decimal("0.10"), tax_amount=Decimal("5"),
                       total=Decimal("105"))
    result = validate_invoice(invoice, FakeInventory(SEED))
    assert result.is_valid
    assert [(i.code, i.field) for i in result.warnings] == [(MISMATCH, "tax_amount")]


def test_due_date_before_invoice_date_is_an_error() -> None:
    invoice = _invoice(_item("WidgetA", 1), invoice_date="2026-02-01", due_date="2026-01-01")
    result = validate_invoice(invoice, FakeInventory(SEED))
    assert [i.code for i in result.errors] == [IssueCode.INVALID_DATE]


def test_missing_required_fields() -> None:
    result = validate_invoice(Invoice(invoice_number="INV-1"), FakeInventory(SEED))
    assert {i.field for i in result.errors} == {"vendor_name", "total", "items"}
    assert {i.field for i in result.warnings} == {"invoice_date", "due_date"}
