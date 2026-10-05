"""Entry point for the ingestion stage: file path -> ``Invoice``."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from invoice_processor.ingestion.builder import RawInvoice, build_invoice
from invoice_processor.ingestion.errors import IngestionError
from invoice_processor.ingestion.pdf import pdf_to_text
from invoice_processor.ingestion.structured import parse_csv, parse_json, parse_xml
from invoice_processor.ingestion.text_parser import parse_text
from invoice_processor.models import Invoice

logger = logging.getLogger(__name__)


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"File is not valid UTF-8 text: {exc}") from exc


_PARSERS: dict[str, Callable[[bytes], RawInvoice]] = {
    ".txt": lambda data: parse_text(_decode(data)),
    ".pdf": lambda data: parse_text(pdf_to_text(data)),
    ".json": lambda data: parse_json(_decode(data)),
    ".csv": lambda data: parse_csv(_decode(data)),
    ".xml": parse_xml,
}
SUPPORTED_EXTENSIONS = frozenset(_PARSERS)


def extract_invoice(path: str | Path) -> Invoice:
    """Extract structured invoice data from a TXT, PDF, JSON, CSV, or XML file.

    Missing or unreadable fields become ``None`` and are explained in
    ``Invoice.extraction_warnings``. No validation is performed.

    Raises:
        IngestionError: If the file is missing, unsupported, malformed, or
            contains no recognizable invoice data.
    """
    path = Path(path)
    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestionError(f"Unsupported file type {path.suffix!r} for {path} (supported: {supported})")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise IngestionError(f"Could not read invoice file {path}: {exc}") from exc

    try:
        invoice = build_invoice(parser(data), source_file=str(path))
    except IngestionError as exc:
        raise IngestionError(f"{path.name}: {exc}") from exc

    logger.info(
        "Extracted %s from %s: %d items, total=%s, %d warnings",
        invoice.invoice_number, path.name, len(invoice.items), invoice.total, len(invoice.extraction_warnings),
    )
    for warning in invoice.extraction_warnings:
        logger.warning("%s: %s", path.name, warning)
    return invoice
