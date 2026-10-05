import sqlite3
from pathlib import Path

import pytest

from invoice_processor.database import InventoryDatabaseError, get_connection, init_db
from invoice_processor.inventory import SQLiteInventory


@pytest.fixture
def inventory(tmp_path: Path) -> SQLiteInventory:
    return SQLiteInventory(init_db(tmp_path / "inventory.db"))


def test_get_stock_levels_returns_known_items_only(inventory: SQLiteInventory) -> None:
    levels = inventory.get_stock_levels(["WidgetA", "GadgetX", "FakeItem", "SuperGizmo", "WidgetA"])
    assert levels == {"WidgetA": 15, "GadgetX": 5, "FakeItem": 0}


def test_get_stock(inventory: SQLiteInventory) -> None:
    assert inventory.get_stock("WidgetB") == 10
    assert inventory.get_stock("WidgetC") is None


def test_lookup_is_case_sensitive(inventory: SQLiteInventory) -> None:
    assert inventory.get_stock("widgeta") is None


def test_empty_lookup_does_not_touch_database(tmp_path: Path) -> None:
    assert SQLiteInventory(tmp_path / "missing.db").get_stock_levels([]) == {}


def test_missing_database_raises(tmp_path: Path) -> None:
    with pytest.raises(InventoryDatabaseError, match="not found"):
        SQLiteInventory(tmp_path / "missing.db").get_stock_levels(["WidgetA"])


def test_missing_table_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()
    with pytest.raises(InventoryDatabaseError, match="lookup failed"):
        SQLiteInventory(path).get_stock_levels(["WidgetA"])


def test_read_only_connection_rejects_writes(inventory: SQLiteInventory) -> None:
    conn = get_connection(inventory.db_path, read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE inventory SET stock = 0")
    finally:
        conn.close()
