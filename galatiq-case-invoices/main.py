"""Invoice processing CLI.

Usage:
    python main.py --invoice_path=data/invoices/
    python main.py --invoice_path=data/invoices/invoice_1001.txt --llm=off
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from invoice_processor.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
