"""Create (or refresh) the local SQLite inventory database.

Usage:
    python scripts/init_db.py               # create/refresh ./inventory.db
    python scripts/init_db.py --reset       # drop and recreate the table
    python scripts/init_db.py --db-path other.db
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_processor.database import (  # noqa: E402
    DEFAULT_DB_PATH,
    InventoryDatabaseError,
    init_db,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or refresh the inventory database.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite file to create (default: {DEFAULT_DB_PATH.name} in the project root)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop the inventory table first, removing any non-seed items",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        init_db(args.db_path, reset=args.reset)
    except InventoryDatabaseError as exc:
        logging.getLogger("init_db").error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
