"""LLM approval agent with a bounded draft -> critique -> revise loop.

    draft_1 = agent(invoice, validation, policy)
    for round in 1..critique_rounds (max 2):
        critique = deterministic checks + LLM critic (given the policy engine's facts)
        if no findings: stop
        draft_n+1 = agent.revise(draft_n, critique)
    final = constrain(draft, deterministic policy)

Constraints on the final decision:
* It is never more lenient than ``RuleBasedApprover``: if the policy engine
  rejects, the result is REJECTED. The agent may be stricter (reject what the
  rules would approve), with its reasoning recorded.
* ``requires_additional_scrutiny`` always comes from the policy, not the agent.
* ``approve_invoice`` still applies ``enforce_hard_rules`` on top.
If the LLM is unavailable, the deterministic policy decision is used and flagged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from invoice_processor.approval import ApprovalPolicy, RuleBasedApprover
from invoice_processor.llm import LLMClient, LLMError
from invoice_processor.models import ApprovalDecision, ApprovalResult, Invoice, ValidationResult

logger = logging.getLogger(__name__)

MAX_CRITIQUE_ROUNDS = 2

FLAG_LLM_REVIEWED = "llm_reviewed"
FLAG_LLM_UNAVAILABLE = "llm_unavailable"
FLAG_OVERRIDDEN_BY_POLICY = "llm_overridden_by_policy"
FLAG_STRICTER_THAN_POLICY = "llm_stricter_than_policy"
FLAG_CRITIQUE_UNRESOLVED = "critique_unresolved"

CritiqueCategory = Literal[
    "missed_validation_issue", "policy_violation", "insufficient_justification", "scrutiny_rule"
]


class ApprovalDraft(BaseModel):
    decision: ApprovalDecision
    requires_additional_scrutiny: bool
    reasoning: str = Field(min_length=1, description="Cite the amounts and every validation issue considered.")


class CritiqueFinding(BaseModel):
    category: CritiqueCategory
    detail: str = Field(min_length=1)


class Critique(BaseModel):
    findings: list[CritiqueFinding] = Field(default_factory=list, description="Empty if the draft is correct.")


@dataclass(frozen=True)
class PolicyFacts:
    """Deterministic ground truth the critic checks drafts against."""

    baseline: ApprovalResult
    usd_amount: Decimal | None
    threshold: Decimal

    @property
    def requires_scrutiny(self) -> bool:
        return self.baseline.requires_additional_scrutiny

    def scrutiny_explanation(self) -> str:
        threshold = f"${self.threshold:,.2f}"
        if self.usd_amount is None:
            return f"The USD value cannot be determined, so it cannot be shown to be under {threshold}: scrutiny IS required."
        amount = f"${self.usd_amount:,.2f}"
        if self.requires_scrutiny:
            return f"{amount} is strictly greater than {threshold}: scrutiny IS required."
        return f"{amount} is not greater than {threshold} (the rule applies only above it): scrutiny is NOT required."


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

AGENT_SYSTEM = """You are the VP-level invoice approval agent at Acme Corp.
Decide whether to APPROVE or REJECT the invoice for payment under the policy below, and justify the
decision by citing the specific amounts and every validation issue you considered. Be conservative:
when policy and evidence conflict, reject for manual review. You cannot pay invoices or change data."""

CRITIC_SYSTEM = """You are an independent reviewer auditing an approval agent's draft decision.
The POLICY ENGINE FACTS are computed deterministically and are authoritative.
Report a finding only for a genuine problem, in one of these categories:
- missed_validation_issue: a validation issue the draft ignores or misreads
- policy_violation: the decision contradicts the policy or the policy engine facts
- insufficient_justification: the reasoning does not support the decision with specifics
- scrutiny_rule: the draft misapplies the $10,000 additional-scrutiny rule
Return no findings if the draft is correct and well justified."""

REVISE_SYSTEM = AGENT_SYSTEM + """
A reviewer has critiqued your previous draft. Produce a revised decision that resolves every finding."""


def _policy_text(policy: ApprovalPolicy) -> str:
    threshold = f"${policy.scrutiny_threshold_usd:,.2f} USD"
    currencies = ", ".join(sorted(policy.usd_exchange_rates)) or "none"
    blocking = ", ".join(sorted(c.value for c in policy.always_blocking)) or "none"
    exempt = ", ".join(sorted(c.value for c in policy.scrutiny_exempt)) or "none"
    return f"""APPROVAL POLICY
- Any validation ERROR: reject.
- A missing, zero, or negative total: reject.
- Additional scrutiny is required when the USD total is strictly greater than {threshold} (exactly
  {threshold} does not require it), or when the USD value cannot be determined (currencies with a
  configured USD rate: {currencies}).
- Warnings that block approval at any amount: {blocking}.
- Under additional scrutiny, every other unresolved WARNING blocks approval (exempt: {exempt}).
- Otherwise warnings are acceptable but must be acknowledged in the reasoning."""


def _context(invoice: Invoice, validation: ValidationResult, policy: ApprovalPolicy) -> str:
    return (
        f"{_policy_text(policy)}\n\nINVOICE\n{invoice.model_dump_json(indent=2, exclude={'source_file'})}\n\n"
        f"VALIDATION RESULT\n{validation.model_dump_json(indent=2, exclude={'validated_at'})}"
    )


def _facts_text(facts: PolicyFacts) -> str:
    return (
        f"POLICY ENGINE FACTS (authoritative)\n- {facts.scrutiny_explanation()}\n"
        f"- Policy engine decision: {facts.baseline.decision.value.upper()}\n"
        f"- Policy engine reasoning:\n{facts.baseline.reasoning}"
    )


# --------------------------------------------------------------------------- #
# Deterministic critique
# --------------------------------------------------------------------------- #


def _mentions(text: str, *terms: str | None) -> bool:
    haystack = text.lower().replace("_", " ")
    return any(t and t.lower().replace("_", " ") in haystack for t in terms)


def deterministic_findings(
    draft: ApprovalDraft, facts: PolicyFacts, validation: ValidationResult, min_reasoning_chars: int
) -> list[CritiqueFinding]:
    findings = []
    if draft.decision is ApprovalDecision.APPROVED and facts.baseline.decision is ApprovalDecision.REJECTED:
        findings.append(CritiqueFinding(
            category="policy_violation",
            detail=f"The policy engine rejects this invoice; approval is not permitted. {facts.baseline.reasoning}",
        ))
    if draft.requires_additional_scrutiny != facts.requires_scrutiny:
        findings.append(CritiqueFinding(category="scrutiny_rule", detail=facts.scrutiny_explanation()))
    unaddressed = [
        i for i in validation.issues
        if not _mentions(draft.reasoning, i.item, i.field, i.code.value)
    ]
    if unaddressed:
        listed = "; ".join(f"[{i.severity.value}/{i.code.value}] {i.message}" for i in unaddressed)
        findings.append(CritiqueFinding(category="missed_validation_issue",
                                        detail=f"The reasoning does not address: {listed}"))
    if len(draft.reasoning.strip()) < min_reasoning_chars:
        findings.append(CritiqueFinding(
            category="insufficient_justification",
            detail=f"Reasoning is too brief ({len(draft.reasoning.strip())} characters); cite the amount, "
                   "the scrutiny rule, and each validation issue.",
        ))
    return findings


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class LLMApprover:
    """``Approver`` that drafts with an LLM, critiques, revises, then applies policy constraints."""

    reviewer = "llm-approval-agent"

    def __init__(
        self,
        llm: LLMClient,
        policy: ApprovalPolicy | None = None,
        *,
        critique_rounds: int = MAX_CRITIQUE_ROUNDS,
        min_reasoning_chars: int = 60,
    ) -> None:
        if not 0 <= critique_rounds <= MAX_CRITIQUE_ROUNDS:
            raise ValueError(f"critique_rounds must be between 0 and {MAX_CRITIQUE_ROUNDS}")
        self.llm = llm
        self.policy = policy or ApprovalPolicy()
        self.critique_rounds = critique_rounds
        self.min_reasoning_chars = min_reasoning_chars

    def review(self, invoice: Invoice, validation: ValidationResult) -> ApprovalResult:
        baseline = RuleBasedApprover(self.policy).review(invoice, validation)  # also checks invoice ids match
        facts = PolicyFacts(baseline, self.policy.usd_amount(invoice), self.policy.scrutiny_threshold_usd)
        context = _context(invoice, validation, self.policy)
        trail: list[str] = []

        try:
            draft = self.llm.complete_structured(AGENT_SYSTEM, context, ApprovalDraft)
            trail.append(_describe_draft(1, draft))
            unresolved = False
            for round_number in range(1, self.critique_rounds + 1):
                critique = self._critique(context, facts, validation, draft)
                trail.append(_describe_critique(round_number, critique))
                if not critique.findings:
                    unresolved = False
                    break
                draft = self._revise(context, draft, critique)
                trail.append(_describe_draft(round_number + 1, draft))
                unresolved = True  # this revision has not been critiqued yet
        except LLMError as exc:
            logger.warning("Approval agent unavailable for %s: %s", invoice.invoice_number, exc)
            return baseline.model_copy(update={
                "flags": [*baseline.flags, FLAG_LLM_UNAVAILABLE],
                "reasoning": f"{baseline.reasoning}\n(LLM approval agent unavailable: {exc}. "
                             "Deterministic policy decision used.)",
                "review_trail": trail,
            })

        return self._finalize(invoice, draft, facts, trail, unresolved)

    def _critique(
        self, context: str, facts: PolicyFacts, validation: ValidationResult, draft: ApprovalDraft
    ) -> Critique:
        findings = deterministic_findings(draft, facts, validation, self.min_reasoning_chars)
        prompt = f"{context}\n\n{_facts_text(facts)}\n\nDRAFT DECISION\n{draft.model_dump_json(indent=2)}"
        llm_findings = self.llm.complete_structured(CRITIC_SYSTEM, prompt, Critique).findings
        seen = {(f.category, f.detail) for f in findings}
        findings.extend(f for f in llm_findings if (f.category, f.detail) not in seen)
        return Critique(findings=findings)

    def _revise(self, context: str, draft: ApprovalDraft, critique: Critique) -> ApprovalDraft:
        prompt = (
            f"{context}\n\nYOUR PREVIOUS DRAFT\n{draft.model_dump_json(indent=2)}\n\n"
            f"REVIEWER FINDINGS\n{critique.model_dump_json(indent=2)}"
        )
        return self.llm.complete_structured(REVISE_SYSTEM, prompt, ApprovalDraft)

    def _finalize(
        self, invoice: Invoice, draft: ApprovalDraft, facts: PolicyFacts, trail: list[str], unresolved: bool
    ) -> ApprovalResult:
        decision = draft.decision
        flags = [*facts.baseline.flags, FLAG_LLM_REVIEWED]
        notes = []
        if draft.decision is ApprovalDecision.APPROVED and facts.baseline.decision is ApprovalDecision.REJECTED:
            decision = ApprovalDecision.REJECTED
            flags.append(FLAG_OVERRIDDEN_BY_POLICY)
            notes.append(f"Final decision constrained to REJECTED by deterministic policy:\n{facts.baseline.reasoning}")
        elif draft.decision is ApprovalDecision.REJECTED and facts.baseline.decision is ApprovalDecision.APPROVED:
            flags.append(FLAG_STRICTER_THAN_POLICY)
        if draft.requires_additional_scrutiny != facts.requires_scrutiny:
            notes.append(f"Scrutiny set by policy: {facts.scrutiny_explanation()}")
        if unresolved:
            flags.append(FLAG_CRITIQUE_UNRESOLVED)

        result = ApprovalResult(
            invoice_number=invoice.invoice_number,
            decision=decision,
            reasoning="\n".join([draft.reasoning, *notes]),
            requires_additional_scrutiny=facts.requires_scrutiny,
            flags=list(dict.fromkeys(flags)),
            reviewer=self.reviewer,
            review_trail=trail,
        )
        logger.info("LLM approval for %s: %s after %d step(s)%s", invoice.invoice_number, decision.value,
                    len(trail), " (overridden by policy)" if FLAG_OVERRIDDEN_BY_POLICY in flags else "")
        return result


def _describe_draft(number: int, draft: ApprovalDraft) -> str:
    scrutiny = "scrutiny" if draft.requires_additional_scrutiny else "no scrutiny"
    return f"Draft {number}: {draft.decision.value.upper()} ({scrutiny}). {draft.reasoning}"


def _describe_critique(number: int, critique: Critique) -> str:
    if not critique.findings:
        return f"Critique {number}: no issues found."
    return f"Critique {number}: " + " | ".join(f"[{f.category}] {f.detail}" for f in critique.findings)
