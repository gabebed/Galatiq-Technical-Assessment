import sqlite3
from pathlib import Path

import pytest

from invoice_processor.database import (
    SEED_INVENTORY,
    InventoryDatabaseError,
    get_connection,
    init_db,
)

EXPECTED = {"WidgetA": 15, "WidgetB": 10, "GadgetX": 5, "FakeItem": 0}


def _rows(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    try:
        return dict(conn.execute("SELECT item, stock FROM inventory"))
    finally:
        conn.close()


def test_init_db_creates_file_schema_and_seed(tmp_path: Path) -> None:
    db_path = init_db(tmp_path / "inventory.db")

    assert db_path.is_file()
    conn = sqlite3.connect(db_path)
    try:
        # PRAGMA table_info rows: (cid, name, type, notnull, default, pk)
        columns = {row[1]: (row[2], row[5]) for row in conn.execute("PRAGMA table_info(inventory)")}
    finally:
        conn.close()
    assert columns == {"item": ("TEXT", 1), "stock": ("INTEGER", 0)}
    assert _rows(db_path) == EXPECTED
    assert dict(SEED_INVENTORY) == EXPECTED


def test_init_db_is_idempotent_and_restores_seed_stock(tmp_path: Path) -> None:
    db_path = init_db(tmp_path / "inventory.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("UPDATE inventory SET stock = 99 WHERE item = 'WidgetA'")
    conn.close()

    init_db(db_path)

    assert _rows(db_path) == EXPECTED


def test_reset_removes_non_seed_items(tmp_path: Path) -> None:
    db_path = init_db(tmp_path / "inventory.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("INSERT INTO inventory VALUES ('Extra', 1)")
    conn.close()

    init_db(db_path, reset=True)

    assert _rows(db_path) == EXPECTED


def test_invalid_seed_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InventoryDatabaseError):
        init_db(tmp_path / "inventory.db", seed={"WidgetA": -1})


def test_get_connection_requires_existing_db(tmp_path: Path) -> None:
    with pytest.raises(InventoryDatabaseError, match="init_db.py"):
        get_connection(tmp_path / "missing.db")

    conn = get_connection(init_db(tmp_path / "inventory.db"))
    try:
        row = conn.execute("SELECT stock FROM inventory WHERE item = ?", ("GadgetX",)).fetchone()
        assert row["stock"] == 5
    finally:
        conn.close()
