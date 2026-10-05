"""Human-readable and JSON rendering of workflow results (plain ASCII for any console)."""

from __future__ import annotations

import textwrap
from decimal import Decimal
from pathlib import Path
from typing import Any

from invoice_processor.models import PaymentStatus, Severity
from invoice_processor.workflow import InvoiceState, PipelineStatus

WIDTH = 100
_RULE = "=" * WIDTH
_STATUS_LABEL = {
    PipelineStatus.COMPLETED: "PAID",
    PipelineStatus.REJECTED: "REJECTED",
    PipelineStatus.FAILED: "FAILED",
}


# LLM text often contains typographic punctuation that legacy Windows code pages mangle.
_ASCII_PUNCTUATION = str.maketrans({
    "—": "-", "–": "-", "‒": "-", "−": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "×": "x", "→": "->", "≤": "<=",
    "≥": ">=", " ": " ", "•": "*",
})


def _plain(text: str) -> str:
    return text.translate(_ASCII_PUNCTUATION)


def status_label(state: InvoiceState) -> str:
    return _STATUS_LABEL.get(state.get("status"), "UNKNOWN")


def _money(amount: Decimal | None, currency: str = "") -> str:
    if amount is None:
        return "-"
    return f"{amount:,.2f}" + (f" {currency}" if currency else "")


def _wrap(text: str, indent: int, hanging: int = 0) -> list[str]:
    """Wrap to WIDTH; continuation lines get ``hanging`` extra spaces (2 for '- ' bullets)."""
    prefix = " " * indent
    lines = []
    for paragraph in text.splitlines() or [""]:
        extra = hanging or (2 if paragraph.startswith("- ") else 0)
        lines.extend(textwrap.wrap(paragraph, WIDTH, initial_indent=prefix,
                                   subsequent_indent=prefix + " " * extra) or [prefix])
    return lines


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def short_reason(state: InvoiceState) -> str:
    """A few words on why an invoice was not paid (empty if it was)."""
    status = state.get("status")
    if status is PipelineStatus.COMPLETED:
        return ""
    validation, approval, payment = state.get("validation"), state.get("approval"), state.get("payment")
    if status is PipelineStatus.REJECTED:
        if validation is not None and validation.errors:
            codes = list(dict.fromkeys(e.code.value for e in validation.errors))
            return ", ".join(codes[:2]) + (" ..." if len(codes) > 2 else "")
        return "rejected at approval" if approval is not None else "rejected"
    if payment is not None and (payment.message or "").startswith("Duplicate payment blocked"):
        return "duplicate payment blocked"
    if "invoice" not in state:
        return "could not read file"
    return "processing error"


def _field(label: str, value: object) -> str:
    return f"  {label:<14}{value if value not in (None, '') else '-'}"


def format_result(path: Path, state: InvoiceState) -> str:
    invoice = state.get("invoice")
    number = invoice.invoice_number if invoice and invoice.invoice_number else "(no invoice ID)"
    title = f"{number}  |  {path.name}"
    out = [_RULE, f"{title}{status_label(state):>{WIDTH - len(title)}}", _RULE]

    # Extraction
    out.append("EXTRACTION")
    if invoice is None:
        out.append("  Not extracted.")
    else:
        cur = invoice.currency
        out += [
            _field("Vendor", invoice.vendor_name),
            _field("Invoice date", invoice.invoice_date),
            _field("Due date", invoice.due_date),
            _field("Terms", invoice.payment_terms),
            _field("Currency", cur),
            "  Items",
        ]
        if not invoice.items:
            out.append("    (none)")
        for item in invoice.items:
            qty = "?" if item.quantity is None else str(item.quantity)
            line = f"    {item.name:<18}{qty:>6} x {_money(item.unit_price):>11}  = {_money(item.computed_total):>12}"
            out.append(line + (f"   ({item.note})" if item.note else ""))
        totals = [f"Subtotal {_money(invoice.subtotal)}", f"Tax {_money(invoice.tax_amount)}"]
        if invoice.shipping is not None:
            totals.append(f"Shipping {_money(invoice.shipping)}")
        totals.append(f"Total {_money(invoice.total, cur)}")
        out.append(_field("Amounts", "   ".join(totals)))
        for warning in invoice.extraction_warnings:
            out += _wrap(f"! {warning}", 2, hanging=2)

    # Validation
    validation = state.get("validation")
    if validation is None:
        out += ["", "VALIDATION", "  Not run."]
    else:
        verdict = "PASSED" if validation.is_valid else "FAILED"
        counts = f"{_plural(len(validation.errors), 'error')}, {_plural(len(validation.warnings), 'warning')}"
        out += ["", f"VALIDATION  {verdict}  ({counts})"]
        for issue in [*validation.errors, *validation.warnings]:
            tag = "ERROR  " if issue.severity is Severity.ERROR else "WARNING"
            out += _wrap(f"{tag} [{issue.code.value}] {issue.message}", 2, hanging=8)
        review = state.get("validation_review")
        if review is not None:
            out += _wrap(f"Agent review: {review.summary}", 2)
            for concern in review.concerns:
                out += _wrap(f"- {concern.item + ': ' if concern.item else ''}{concern.concern}", 4)

    # Approval
    approval = state.get("approval")
    out += ["", "APPROVAL"]
    if approval is None:
        out.append("  Not reached: invoice failed validation." if validation is not None else "  Not reached.")
    else:
        scrutiny = "additional scrutiny required" if approval.requires_additional_scrutiny else "standard review"
        out[-1] = f"APPROVAL  {approval.decision.value.upper()}  ({scrutiny}; reviewer: {approval.reviewer})"
        out += _wrap(approval.reasoning, 2)
        if approval.flags:
            out += _wrap(f"Flags: {', '.join(approval.flags)}", 2)
        if approval.review_trail:
            out.append(f"  Review trail ({len(approval.review_trail)} steps):")
            for step in approval.review_trail:
                out.append("    " + (step if len(step) <= WIDTH - 4 else step[: WIDTH - 7] + "..."))

    # Payment
    payment = state.get("payment")
    out += ["", "PAYMENT"]
    if payment is not None and payment.status is PaymentStatus.PAID:
        out[-1] = f"PAYMENT  PAID  {_money(payment.amount, payment.currency)} to {payment.vendor_name}"
        out.append(_field("Transaction", payment.transaction_id))
    elif payment is not None:
        out[-1] = f"PAYMENT  {'NOT PAID - ' + payment.status.value.upper()}"
        out += _wrap(payment.message or "", 2)
    elif state.get("status") is PipelineStatus.REJECTED:
        out[-1] = "PAYMENT  SKIPPED"
        stage = "approval" if approval is not None else "validation"
        out.append(f"  Payment was not attempted: the invoice was rejected at {stage}.")
    else:
        out[-1] = "PAYMENT  SKIPPED"
        out.append("  Payment was not attempted: processing did not reach the payment stage.")

    calls = state.get("tool_calls") or []
    if calls:
        refused = sum(not c.ok for c in calls)
        out += ["", f"AGENT TOOL CALLS  {len(calls)} ({refused} refused)"]
        for call in calls:
            outcome = "ok" if call.ok else f"refused: {call.error}"
            out += _wrap(f"{call.toolset}.{call.tool}({call.arguments}) -> {outcome}", 2)

    if state.get("error") and not (payment is not None and state["error"] == payment.message):
        out += ["", "ERROR"] + _wrap(state["error"], 2)
    for error in state.get("agent_errors") or []:
        out += _wrap(f"Agent unavailable: {error}", 2)
    return _plain("\n".join(out))


def format_summary(results: list[tuple[Path, InvoiceState]]) -> str:
    counts = {label: 0 for label in ("PAID", "REJECTED", "FAILED")}
    rows = []
    for path, state in results:
        label = status_label(state)
        counts[label] = counts.get(label, 0) + 1
        invoice = state.get("invoice")
        number = invoice.invoice_number if invoice and invoice.invoice_number else "-"
        total = _money(invoice.total, invoice.currency) if invoice else "-"
        rows.append(f"  {number:<12}{path.name:<28}{label:<10}{total:>16}   {short_reason(state)}".rstrip())
    head = (f"SUMMARY  {len(results)} invoice(s): {counts['PAID']} paid, {counts['REJECTED']} rejected, "
            f"{counts['FAILED']} failed")
    columns = f"  {'Invoice':<12}{'File':<28}{'Result':<10}{'Total':>16}   Reason"
    return _plain("\n".join([_RULE, head, _RULE, columns, *rows]))


def to_json(path: Path, state: InvoiceState) -> dict[str, Any]:
    def dump(key: str) -> Any:
        value = state.get(key)
        return value.model_dump(mode="json") if value is not None else None

    return {
        "file": str(path),
        "result": status_label(state),
        "invoice": dump("invoice"),
        "validation": dump("validation"),
        "validation_review": dump("validation_review"),
        "approval": dump("approval"),
        "payment": dump("payment"),
        "rejection_reason": state.get("rejection_reason"),
        "error": state.get("error"),
        "agent_errors": state.get("agent_errors") or [],
        "tool_calls": [c.model_dump(mode="json") for c in state.get("tool_calls") or []],
    }
