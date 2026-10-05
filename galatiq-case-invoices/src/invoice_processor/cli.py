"""Command-line interface: python main.py --invoice_path=<file or directory>"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path

from invoice_processor.database import DEFAULT_DB_PATH
from invoice_processor.ingestion import SUPPORTED_EXTENSIONS
from invoice_processor.inventory import SQLiteInventory
from invoice_processor.llm import LLMClient, LLMError, create_llm_client
from invoice_processor.llm.telemetry import collect_llm_calls
from invoice_processor.llm.telemetry import logger as telemetry_logger
from invoice_processor.observability import configure_logging, invoice_context
from invoice_processor.report import format_result, format_summary, llm_fallbacks, status_label, to_json
from invoice_processor.workflow import InvoiceState, PipelineStatus, build_workflow, run_invoice

logger = logging.getLogger("invoice_processor.cli")


class ExitCode(IntEnum):
    ALL_PAID = 0
    SOME_REJECTED = 1
    USAGE_ERROR = 2
    SOME_FAILED = 3
    UNEXPECTED_ERROR = 4
    INTERRUPTED = 130


EPILOG = """exit codes:
  0  every invoice was approved and paid
  1  processing completed; at least one invoice was rejected
  2  invalid arguments or setup (bad path, no invoices, missing database or API key)
  3  at least one invoice could not be processed (unreadable file, payment failure, ...)
  4  unexpected internal error
"""


class InputError(Exception):
    """Invalid user input or setup; reported without a traceback."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py", description="Process invoices: extract, validate, approve, and pay.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--invoice_path", "--invoice-path", dest="invoice_path", required=True, type=Path,
                        help="an invoice file, or a directory of invoice files")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH,
                        help=f"SQLite inventory database (default: {DEFAULT_DB_PATH.name} in the project root)")
    parser.add_argument("--llm", choices=("auto", "on", "off"), default="auto",
                        help="use LLM agents: 'auto' uses them when XAI_API_KEY is set (default: auto)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a report")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="log to stderr (-v info, -vv debug)")
    parser.add_argument("--log-format", choices=("text", "json"), default="text",
                        help="stderr log format; 'json' emits one JSON object per line (default: text)")
    parser.add_argument("--llm-log", action=argparse.BooleanOptionalAction, default=True,
                        help="show every LLM request: a start/end line on stderr and an LLM CALLS table per "
                             "invoice. --no-llm-log hides successful requests; failures are always shown "
                             "(default: on)")
    return parser


def collect_invoices(path: Path) -> list[Path]:
    if not path.exists():
        raise InputError(f"Invoice path does not exist: {path}")
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise InputError(f"Unsupported invoice file type {path.suffix!r} (supported: {supported})")
        return [path]
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise InputError(f"No invoice files ({supported}) found in {path}")
    return files


def _select_llm(mode: str) -> LLMClient | None:
    if mode == "off" or (mode == "auto" and not os.environ.get("XAI_API_KEY", "").strip()):
        return None
    try:
        return create_llm_client()
    except LLMError as exc:
        if mode == "on":
            raise InputError(str(exc)) from exc
        # Never fall back silently: say so even when logging is off.
        print(f"warning: LLM agents disabled, running deterministically: {exc}", file=sys.stderr)
        return None


def _exit_code(states: Sequence[InvoiceState]) -> ExitCode:
    statuses = [s.get("status") for s in states]
    if any(s is not PipelineStatus.COMPLETED and s is not PipelineStatus.REJECTED for s in statuses):
        return ExitCode.SOME_FAILED
    if PipelineStatus.REJECTED in statuses:
        return ExitCode.SOME_REJECTED
    return ExitCode.ALL_PAID


def _configure_output(verbosity: int, log_format: str) -> None:
    # Errors are already rendered in the report; logs are opt-in to avoid duplicate output.
    level = logging.CRITICAL if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    configure_logging(level, log_format)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # never crash on characters the console cannot show


def _warn_about_llm_failures(results: list[tuple[Path, InvoiceState]]) -> None:
    """Printed regardless of verbosity: LLM failures must never be silent."""
    calls = [c for _, state in results for c in state.get("llm_calls") or []]
    failed = [c for c in calls if not c.success]
    affected = [path.name for path, state in results if llm_fallbacks(state)]
    if failed or affected:
        print(f"warning: {len(failed)} of {len(calls)} LLM request(s) failed; "
              f"fallback or failure in {len(affected)} invoice(s)"
              + (f": {', '.join(affected)}" if affected else "")
              + ". See the LLM CALLS / LLM FAILURES sections.", file=sys.stderr)


def run(args: argparse.Namespace) -> ExitCode:
    files = collect_invoices(args.invoice_path)
    if not args.db_path.is_file():
        raise InputError(f"Inventory database not found at {args.db_path}. Run: python scripts/init_db.py")

    llm = _select_llm(args.llm)
    # The client wrapper logs every LLM request. With --llm-log those lines are shown even without -v;
    # with --no-llm-log only failed requests (ERROR) are shown, so failures are never silent.
    telemetry_logger.setLevel(logging.INFO if args.llm_log else logging.ERROR)
    # The workflow's default payment function refuses to pay an invoice number twice in one run.
    workflow = build_workflow(SQLiteInventory(args.db_path), llm=llm)
    if llm is None:
        mode = "deterministic (no LLM)"
    else:
        mode = f"LLM agents ({llm!r}; request logging {'on' if args.llm_log else 'off, failures only'})"

    results: list[tuple[Path, InvoiceState]] = []
    logger.info("Run started: %d invoice(s), mode=%s, inventory=%s", len(files), mode, args.db_path)
    if not args.json:
        print(f"Processing {len(files)} invoice(s) | mode: {mode} | inventory: {args.db_path}\n")
    for path in files:
        started = time.perf_counter()
        with invoice_context(path.name), collect_llm_calls() as llm_calls:
            try:
                state = run_invoice(path, workflow)
            except Exception as exc:  # isolate failures: one bad invoice must not stop the batch
                logger.exception("Unexpected error processing %s", path)
                state = {"status": PipelineStatus.FAILED, "error": f"Unexpected error: {type(exc).__name__}: {exc}"}
            state["llm_calls"] = list(llm_calls)
            logger.info("Finished: %s in %.0f ms", status_label(state), (time.perf_counter() - started) * 1000)
        results.append((path, state))
        if not args.json:
            print(format_result(path, state, show_llm_calls=args.llm_log), end="\n\n", flush=True)

    if args.json:
        print(json.dumps({"mode": mode, "invoices": [to_json(p, s) for p, s in results]}, indent=2))
    elif len(results) > 1:
        print(format_summary(results, show_llm_calls=args.llm_log))
    _warn_about_llm_failures(results)
    exit_code = _exit_code([s for _, s in results])
    counts = Counter(status_label(s) for _, s in results)
    logger.info("Run finished: %s; exit code %d", dict(counts), exit_code)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _configure_output(args.verbose, args.log_format)
    try:
        return run(args)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return ExitCode.USAGE_ERROR
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return ExitCode.INTERRUPTED
    except Exception as exc:
        logger.debug("Unexpected error", exc_info=True)
        print(f"unexpected error: {type(exc).__name__}: {exc}" + ("" if args.verbose else " (use -v for details)"),
              file=sys.stderr)
        return ExitCode.UNEXPECTED_ERROR
