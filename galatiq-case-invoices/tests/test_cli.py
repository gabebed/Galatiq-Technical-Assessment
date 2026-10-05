import json
import subprocess
import sys
from pathlib import Path

import pytest

from invoice_processor import cli
from invoice_processor.cli import ExitCode, main
from invoice_processor.database import init_db
from invoice_processor.report import _plain

ROOT = Path(__file__).resolve().parents[1]
INVOICES = ROOT / "data" / "invoices"


@pytest.fixture
def db(tmp_path: Path) -> str:
    return str(init_db(tmp_path / "inventory.db"))


def _run(capsys: pytest.CaptureFixture, *args: str) -> tuple[int, str, str]:
    code = main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


# --------------------------------------------------------------------------- #
# Outcomes and exit codes
# --------------------------------------------------------------------------- #


def test_paid_invoice(capsys, db) -> None:
    code, out, _ = _run(capsys, f"--invoice_path={INVOICES / 'invoice_1001.txt'}", f"--db-path={db}", "--llm=off")

    assert code == ExitCode.ALL_PAID
    for expected in ("INV-1001  |  invoice_1001.txt", "Vendor        Widgets Inc.", "VALIDATION  PASSED",
                     "APPROVAL  APPROVED", "PAYMENT  PAID  5,000.00 USD to Widgets Inc.", "Transaction   MOCK-"):
        assert expected in out
    assert "SUMMARY" not in out  # single invoice: no summary table


def test_rejected_invoice_shows_issues_and_skipped_payment(capsys, db) -> None:
    code, out, _ = _run(capsys, f"--invoice_path={INVOICES / 'invoice_1002.txt'}", f"--db-path={db}", "--llm=off")

    assert code == ExitCode.SOME_REJECTED
    assert "VALIDATION  FAILED  (1 error, 1 warning)" in out
    assert "ERROR   [insufficient_stock] Invoice bills 20 units of GadgetX" in out
    assert "Not reached: invoice failed validation." in out
    assert "PAYMENT  SKIPPED" in out and "rejected at validation" in out


def test_directory_run_with_summary_and_duplicate_protection(capsys, db) -> None:
    code, out, _ = _run(capsys, f"--invoice_path={INVOICES}", f"--db-path={db}", "--llm=off")

    assert code == ExitCode.SOME_FAILED  # duplicate copies need manual reconciliation
    assert "SUMMARY  20 invoice(s): 8 paid, 9 rejected, 3 failed" in out
    assert out.count("duplicate payment blocked") == 3
    assert "Duplicate payment blocked: invoice INV-1004 was already paid 1,890.00 USD" in out
    assert out.count("PAYMENT  PAID") == 8


def test_json_output(capsys, db) -> None:
    code, out, _ = _run(capsys, f"--invoice_path={INVOICES / 'invoice_1001.txt'}", f"--db-path={db}",
                        "--llm=off", "--json")
    data = json.loads(out)

    assert code == ExitCode.ALL_PAID
    [result] = data["invoices"]
    assert result["result"] == "PAID"
    assert result["invoice"]["invoice_number"] == "INV-1001"
    assert result["payment"]["status"] == "paid"
    assert result["approval"]["decision"] == "approved"


def test_unreadable_file_in_directory_does_not_stop_batch(capsys, db, tmp_path) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "invoice_1001.txt").write_bytes((INVOICES / "invoice_1001.txt").read_bytes())
    (batch / "broken.pdf").write_bytes((INVOICES / "invoice_1011.pdf").read_bytes()[:300])

    code, out, err = _run(capsys, f"--invoice_path={batch}", f"--db-path={db}", "--llm=off")

    assert code == ExitCode.SOME_FAILED
    assert "1 paid, 0 rejected, 1 failed" in out
    assert "could not read file" in out
    assert err == ""  # errors are in the report, not duplicated as logs


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("path", "message"),
    [("does/not/exist", "does not exist"), (str(ROOT / "README.md"), "Unsupported invoice file type")],
)
def test_invalid_invoice_path(capsys, db, path: str, message: str) -> None:
    code, _, err = _run(capsys, f"--invoice_path={path}", f"--db-path={db}")
    assert code == ExitCode.USAGE_ERROR
    assert message in err


def test_directory_without_invoices(capsys, db, tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    code, _, err = _run(capsys, f"--invoice_path={tmp_path / 'empty'}", f"--db-path={db}")
    assert code == ExitCode.USAGE_ERROR
    assert "No invoice files" in err


def test_missing_database(capsys, tmp_path) -> None:
    code, _, err = _run(capsys, f"--invoice_path={INVOICES / 'invoice_1001.txt'}", f"--db-path={tmp_path / 'x.db'}")
    assert code == ExitCode.USAGE_ERROR
    assert "scripts/init_db.py" in err


def test_llm_on_without_api_key(capsys, db, monkeypatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    code, _, err = _run(capsys, f"--invoice_path={INVOICES / 'invoice_1001.txt'}", f"--db-path={db}", "--llm=on")
    assert code == ExitCode.USAGE_ERROR
    assert "XAI_API_KEY is not set" in err


def test_llm_auto_without_api_key_runs_deterministically(capsys, db, monkeypatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    code, out, _ = _run(capsys, f"--invoice_path={INVOICES / 'invoice_1001.txt'}", f"--db-path={db}")
    assert code == ExitCode.ALL_PAID
    assert "mode: deterministic (no LLM)" in out


def test_missing_required_argument(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == ExitCode.USAGE_ERROR


# --------------------------------------------------------------------------- #
# Unexpected errors
# --------------------------------------------------------------------------- #


def test_unexpected_error_in_one_invoice_is_isolated(capsys, db, monkeypatch) -> None:
    def explode(path, workflow):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "run_invoice", explode)
    code, out, _ = _run(capsys, f"--invoice_path={INVOICES / 'invoice_1001.txt'}", f"--db-path={db}", "--llm=off")

    assert code == ExitCode.SOME_FAILED
    assert "Unexpected error: RuntimeError: kaboom" in out
    assert "PAYMENT  SKIPPED" in out


def test_unexpected_top_level_error(capsys, db, monkeypatch) -> None:
    monkeypatch.setattr(cli, "collect_invoices", lambda path: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    code, _, err = _run(capsys, f"--invoice_path={INVOICES}", f"--db-path={db}")

    assert code == ExitCode.UNEXPECTED_ERROR
    assert "unexpected error: RuntimeError: disk on fire (use -v for details)" in err


def test_keyboard_interrupt(capsys, db, monkeypatch) -> None:
    def interrupt(path, workflow):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_invoice", interrupt)
    code, _, err = _run(capsys, f"--invoice_path={INVOICES / 'invoice_1001.txt'}", f"--db-path={db}", "--llm=off")
    assert code == ExitCode.INTERRUPTED
    assert "Interrupted" in err


# --------------------------------------------------------------------------- #
# Entry point and output encoding
# --------------------------------------------------------------------------- #


def test_main_py_entry_point(db) -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), f"--invoice_path={INVOICES / 'invoice_1016.json'}",
         f"--db-path={db}", "--llm=off"],
        capture_output=True, text=True, cwd=ROOT, timeout=60,
    )
    assert proc.returncode == ExitCode.SOME_REJECTED
    assert "WidgetC is not in the inventory database" in proc.stdout


def test_llm_punctuation_is_made_ascii() -> None:
    assert _plain("OCR — “corrected” 7 × $500…") == 'OCR - "corrected" 7 x $500...'
