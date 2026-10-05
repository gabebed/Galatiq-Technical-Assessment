import json
import logging
from pathlib import Path

import pytest

from invoice_processor.cli import main
from invoice_processor.database import init_db
from invoice_processor.observability import InvoiceContextFilter, JsonFormatter, invoice_context

INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"


@pytest.fixture(autouse=True)
def _restore_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _record(message: str) -> logging.LogRecord:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, message, None, None)
    InvoiceContextFilter().filter(record)
    return record


def test_records_are_tagged_with_the_current_invoice() -> None:
    assert _record("outside").invoice == "-"
    with invoice_context("invoice_1001.txt"):
        assert _record("inside").invoice == "invoice_1001.txt"
    assert _record("after").invoice == "-"


def test_json_formatter_emits_one_parseable_object() -> None:
    with invoice_context("invoice_1002.txt"):
        entry = json.loads(JsonFormatter().format(_record("Validated INV-1002")))
    assert entry["invoice"] == "invoice_1002.txt"
    assert entry["level"] == "INFO" and entry["message"] == "Validated INV-1002"
    assert entry["ts"].endswith("+00:00")


def test_cli_json_logs_trace_each_stage(capsys, tmp_path) -> None:
    db = init_db(tmp_path / "inventory.db")
    code = main([f"--invoice_path={INVOICES / 'invoice_1001.txt'}", f"--db-path={db}", "--llm=off",
                 "-v", "--log-format=json"])
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]

    assert code == 0
    tagged = [entry["logger"] for entry in lines if entry["invoice"] == "invoice_1001.txt"]
    for stage in ("ingestion.extract", "validation", "approval", "payment", "cli"):
        assert any(name.endswith(stage) for name in tagged), stage
    assert lines[-1]["message"].startswith("Run finished: {'PAID': 1}; exit code 0")
