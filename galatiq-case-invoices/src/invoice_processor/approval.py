"""Approval stage: decide whether a validated invoice may be paid.

Structure:

* ``ApprovalPolicy`` - business rules as data (threshold, FX rates, blocking warnings).
* ``Approver`` - the decision-maker interface. ``RuleBasedApprover`` is the
  deterministic implementation; an LLM-based VP agent can implement the same
  interface later, or wrap this one and refine its reasoning.
* ``approve_invoice`` - entry point. Runs any ``Approver`` and then enforces the
  non-negotiable rules, so no approver (deterministic or LLM) can approve an
  invoice with hard failures or skip required scrutiny.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Protocol

from invoice_processor.models import (
    ApprovalDecision,
    ApprovalResult,
    Invoice,
    IssueCode,
    ValidationIssue,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# Flags recorded on ApprovalResult.flags
FLAG_VALIDATION_FAILED = "validation_failed"
FLAG_INVALID_AMOUNT = "invalid_amount"
FLAG_OVER_THRESHOLD = "over_threshold"
FLAG_CURRENCY_NOT_CONVERTED = "currency_not_converted"
FLAG_BLOCKING_WARNINGS = "blocking_warnings"
FLAG_OVERRIDDEN = "approval_overridden"


@dataclass(frozen=True)
class ApprovalPolicy:
    scrutiny_threshold_usd: Decimal = Decimal("10000")
    # Rates to convert invoice currency to USD. Unknown currencies always get scrutiny,
    # because the invoice cannot be shown to be under the threshold.
    usd_exchange_rates: Mapping[str, Decimal] = field(
        default_factory=lambda: MappingProxyType({"USD": Decimal(1)})
    )
    # Warnings that block approval at any amount.
    always_blocking: frozenset[IssueCode] = frozenset({IssueCode.SUSPICIOUS_CONTENT})
    # Warnings that do not block approval even under scrutiny.
    scrutiny_exempt: frozenset[IssueCode] = frozenset({IssueCode.UNSUPPORTED_CURRENCY})

    def usd_amount(self, invoice: Invoice) -> Decimal | None:
        rate = self.usd_exchange_rates.get(invoice.currency)
        if invoice.total is None or rate is None:
            return None
        return invoice.total * rate

    def requires_scrutiny(self, invoice: Invoice) -> bool:
        """Over the threshold (strictly greater), or USD value unknown."""
        amount = self.usd_amount(invoice)
        return amount is None or amount > self.scrutiny_threshold_usd


class Approver(Protocol):
    def review(self, invoice: Invoice, validation: ValidationResult) -> ApprovalResult: ...


def _usd(amount: Decimal) -> str:
    return f"${amount:,.2f}"


def _check_same_invoice(invoice: Invoice, validation: ValidationResult) -> None:
    if validation.invoice_number != invoice.invoice_number:
        raise ValueError(
            f"ValidationResult is for {validation.invoice_number!r}, not invoice {invoice.invoice_number!r}"
        )


def hard_failures(invoice: Invoice, validation: ValidationResult) -> list[str]:
    """Reasons an invoice must be rejected regardless of who is approving it.

    The amount check repeats validation on purpose: approval must not trust
    that validation ran or was complete.
    """
    reasons = [issue.message for issue in validation.errors]
    total_already_flagged = any(issue.field == "total" for issue in validation.errors)
    if not total_already_flagged:
        if invoice.total is None:
            reasons.append("Invoice has no stated total to approve.")
        elif invoice.total <= 0:
            reasons.append(f"Total {invoice.total:,.2f} {invoice.currency} is not a positive amount.")
    return reasons


class RuleBasedApprover:
    """Deterministic VP-approval rules."""

    reviewer = "rule-based-approver"

    def __init__(self, policy: ApprovalPolicy | None = None) -> None:
        self.policy = policy or ApprovalPolicy()

    def review(self, invoice: Invoice, validation: ValidationResult) -> ApprovalResult:
        _check_same_invoice(invoice, validation)
        policy = self.policy
        scrutiny = policy.requires_scrutiny(invoice)
        flags: list[str] = []
        reasoning: list[str] = []

        # 1. Hard failures: never approvable.
        failures = hard_failures(invoice, validation)
        if failures:
            flags.append(FLAG_VALIDATION_FAILED if validation.errors else FLAG_INVALID_AMOUNT)
            if invoice.total is None or invoice.total <= 0:
                flags.append(FLAG_INVALID_AMOUNT)
            reasoning.append(f"Rejected: {len(failures)} hard failure(s) must be resolved before approval:")
            reasoning.extend(f"- {reason}" for reason in failures)
            return self._result(invoice, ApprovalDecision.REJECTED, reasoning, scrutiny, flags)

        # 2. Amount and scrutiny.
        usd = policy.usd_amount(invoice)
        threshold = _usd(policy.scrutiny_threshold_usd)
        if usd is None:
            flags.append(FLAG_CURRENCY_NOT_CONVERTED)
            reasoning.append(
                f"Total {invoice.total:,.2f} {invoice.currency} cannot be converted to USD (no rate configured), "
                f"so it cannot be shown to be under the {threshold} threshold; additional scrutiny applies."
            )
        elif scrutiny:
            flags.append(FLAG_OVER_THRESHOLD)
            reasoning.append(f"Total {_usd(usd)} exceeds the {threshold} threshold; additional scrutiny applies.")
        else:
            reasoning.append(f"Total {_usd(usd)} is within the {threshold} threshold; standard review applies.")

        # 3. Warnings: some always block; under scrutiny, all non-exempt warnings block.
        warnings = validation.warnings
        flags.extend(f"warning:{code.value}" for code in dict.fromkeys(w.code for w in warnings))
        blocking = [
            w for w in warnings
            if w.code in policy.always_blocking or (scrutiny and w.code not in policy.scrutiny_exempt)
        ]
        if blocking:
            flags.append(FLAG_BLOCKING_WARNINGS)
            reasoning.append(f"Rejected: {len(blocking)} unresolved warning(s) block approval and need manual review:")
            reasoning.extend(f"- {_describe(w)}" for w in blocking)
            return self._result(invoice, ApprovalDecision.REJECTED, reasoning, scrutiny, flags)

        if warnings:
            reasoning.append(f"{len(warnings)} non-blocking warning(s) noted:")
            reasoning.extend(f"- {_describe(w)}" for w in warnings)
        else:
            reasoning.append("Validation passed with no issues.")
        reasoning.append("Approved for payment.")
        return self._result(invoice, ApprovalDecision.APPROVED, reasoning, scrutiny, flags)

    def _result(
        self, invoice: Invoice, decision: ApprovalDecision, reasoning: list[str], scrutiny: bool, flags: list[str]
    ) -> ApprovalResult:
        return ApprovalResult(
            invoice_number=invoice.invoice_number,
            decision=decision,
            reasoning="\n".join(reasoning),
            requires_additional_scrutiny=scrutiny,
            flags=list(dict.fromkeys(flags)),
            reviewer=self.reviewer,
        )


def _describe(issue: ValidationIssue) -> str:
    return f"[{issue.code.value}] {issue.message}"


def enforce_hard_rules(
    invoice: Invoice, validation: ValidationResult, result: ApprovalResult, policy: ApprovalPolicy
) -> ApprovalResult:
    """Correct any approver output that violates non-negotiable rules."""
    updates: dict[str, object] = {}
    failures = hard_failures(invoice, validation)
    if result.is_approved and failures:
        logger.error("%s: approver %s approved an invoice with hard failures; overriding",
                     invoice.invoice_number, result.reviewer)
        updates["decision"] = ApprovalDecision.REJECTED
        updates["reasoning"] = "\n".join(
            [result.reasoning, "Overridden to REJECTED: hard failures cannot be approved:"]
            + [f"- {reason}" for reason in failures]
        )
        updates["flags"] = [*result.flags, FLAG_OVERRIDDEN]
    if policy.requires_scrutiny(invoice) and not result.requires_additional_scrutiny:
        updates["requires_additional_scrutiny"] = True
    if result.invoice_number != invoice.invoice_number:
        updates["invoice_number"] = invoice.invoice_number
    return result.model_copy(update=updates) if updates else result


def approve_invoice(
    invoice: Invoice,
    validation: ValidationResult,
    approver: Approver | None = None,
    policy: ApprovalPolicy | None = None,
) -> ApprovalResult:
    """Run the approver, then enforce hard rules on its decision.

    Raises:
        ValueError: If ``validation`` belongs to a different invoice.
    """
    _check_same_invoice(invoice, validation)
    policy = policy or ApprovalPolicy()
    approver = approver or RuleBasedApprover(policy)
    result = enforce_hard_rules(invoice, validation, approver.review(invoice, validation), policy)
    logger.info(
        "Approval for %s: %s (scrutiny=%s, flags=%s)",
        invoice.invoice_number, result.decision.value, result.requires_additional_scrutiny, result.flags,
    )
    return result
