"""PDF text extraction via pdfplumber."""

from __future__ import annotations

import io

import pdfplumber

from invoice_processor.ingestion.errors import IngestionError


def pdf_to_text(data: bytes) -> str:
    """Return the text of every page, joined by newlines.

    Raises:
        IngestionError: If the PDF is corrupt or contains no text layer.
    """
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:  # pdfplumber/pdfminer raise many unrelated exception types
        raise IngestionError(f"Could not read PDF: {exc}") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise IngestionError("PDF has no extractable text (scanned image? OCR is not supported)")
    return text
