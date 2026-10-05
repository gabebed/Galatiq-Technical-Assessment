"""SQLite inventory database: schema, seed data, and connection helpers.

The validation stage checks extracted invoice line items against this
database. Initialization is idempotent: running it repeatedly always
leaves the seed items at their canonical stock levels.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH: Final[Path] = PROJECT_ROOT / "inventory.db"

SEED_INVENTORY: Final[Mapping[str, int]] = {
    "WidgetA": 15,
    "WidgetB": 10,
    "GadgetX": 5,
    "FakeItem": 0,
}

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS inventory (
    item  TEXT PRIMARY KEY,
    stock INTEGER NOT NULL CHECK (stock >= 0)
)
"""

_UPSERT: Final[str] = """
INSERT INTO inventory (item, stock) VALUES (?, ?)
ON CONFLICT(item) DO UPDATE SET stock = excluded.stock
"""


class InventoryDatabaseError(RuntimeError):
    """Raised when the inventory database cannot be created or accessed."""


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection to an existing inventory database.

    Raises:
        InventoryDatabaseError: If the file does not exist or cannot be opened.
    """
    path = Path(db_path)
    if not path.is_file():
        raise InventoryDatabaseError(
            f"Inventory database not found at {path}. "
            "Run `python scripts/init_db.py` to create it."
        )
    try:
        conn = sqlite3.connect(path)
    except sqlite3.Error as exc:
        raise InventoryDatabaseError(f"Could not open database at {path}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    return conn


def init_db(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    reset: bool = False,
    seed: Mapping[str, int] = SEED_INVENTORY,
) -> Path:
    """Create the inventory table and load seed data.

    Args:
        db_path: Location of the SQLite file; parent directories are created.
        reset: Drop the existing table first, removing any non-seed items.
        seed: Item -> stock mapping to load. Existing rows are overwritten.

    Returns:
        The resolved path of the database file.

    Raises:
        InventoryDatabaseError: On invalid seed data or any SQLite failure.
    """
    _validate_seed(seed)
    path = Path(db_path).resolve()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
    except (OSError, sqlite3.Error) as exc:
        raise InventoryDatabaseError(f"Could not create database at {path}: {exc}") from exc

    try:
        with conn:  # single transaction: commits on success, rolls back on error
            if reset:
                conn.execute("DROP TABLE IF EXISTS inventory")
            conn.execute(_SCHEMA)
            conn.executemany(_UPSERT, seed.items())
    except sqlite3.Error as exc:
        raise InventoryDatabaseError(f"Failed to initialize database at {path}: {exc}") from exc
    finally:
        conn.close()

    logger.info("Initialized inventory database at %s with %d seed items", path, len(seed))
    return path


def _validate_seed(seed: Mapping[str, int]) -> None:
    for item, stock in seed.items():
        if not isinstance(item, str) or not item.strip():
            raise InventoryDatabaseError(f"Invalid item name in seed data: {item!r}")
        if isinstance(stock, bool) or not isinstance(stock, int) or stock < 0:
            raise InventoryDatabaseError(f"Invalid stock for {item!r}: {stock!r}")
