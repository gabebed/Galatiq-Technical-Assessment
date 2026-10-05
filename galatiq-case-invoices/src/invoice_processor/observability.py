"""Logging setup: every record is tagged with the invoice being processed.

    with invoice_context("invoice_1001.txt"):
        ...  # all log records here carry invoice="invoice_1001.txt"

``configure_logging(fmt="json")`` emits one JSON object per line for log
aggregation; ``fmt="text"`` is for humans.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

_current_invoice: ContextVar[str] = ContextVar("current_invoice", default="-")

TEXT_FORMAT = "%(asctime)s %(levelname)-7s [%(invoice)s] %(name)s: %(message)s"


@contextmanager
def invoice_context(label: str) -> Iterator[None]:
    token = _current_invoice.set(label)
    try:
        yield
    finally:
        _current_invoice.reset(token)


class InvoiceContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.invoice = _current_invoice.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "invoice": getattr(record, "invoice", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: int, fmt: str = "text") -> None:
    """Send logs to stderr (stdout is reserved for the report / JSON results)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(InvoiceContextFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
