"""Read-only inventory lookup used by deterministic validation and the validation agent's tools."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from invoice_processor.database import DEFAULT_DB_PATH, InventoryDatabaseError, get_connection


class InventoryLookup(Protocol):
    def get_stock_levels(self, items: Iterable[str]) -> dict[str, int]:
        """Return {item: stock} for the items that exist; unknown items are omitted."""
        ...


class SQLiteInventory:
    """Read-only access to the SQLite inventory table."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)

    def get_stock_levels(self, items: Iterable[str]) -> dict[str, int]:
        """Look up stock for many items in a single query.

        Raises:
            InventoryDatabaseError: If the database is missing or the query fails.
        """
        names = sorted(set(items))
        if not names:
            return {}
        placeholders = ", ".join("?" for _ in names)
        conn = get_connection(self.db_path, read_only=True)
        try:
            rows = conn.execute(
                f"SELECT item, stock FROM inventory WHERE item IN ({placeholders})", names
            ).fetchall()
        except sqlite3.Error as exc:
            raise InventoryDatabaseError(f"Inventory lookup failed ({self.db_path}): {exc}") from exc
        finally:
            conn.close()

        levels = {}
        for row in rows:
            stock = row["stock"]
            # Corrupt stock (NULL, text, negative) must fail closed, never be guessed at.
            if isinstance(stock, bool) or not isinstance(stock, int) or stock < 0:
                raise InventoryDatabaseError(
                    f"Invalid stock value {stock!r} for {row['item']!r} in {self.db_path}; "
                    "stock must be a non-negative integer."
                )
            levels[row["item"]] = stock
        return levels

    def get_stock(self, item: str) -> int | None:
        """Stock for one item, or None if it is not in inventory."""
        return self.get_stock_levels([item]).get(item)
