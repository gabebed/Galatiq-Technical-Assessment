"""Reflective approval loop: critiques correct bad drafts; policy constrains the final decision."""

from decimal import Decimal
from pathlib import Path

import pytest

from invoice_processor.approval import FLAG_BLOCKING_WARNINGS, approve_invoice
from invoice_processor.approval_agent import (
    FLAG_CRITIQUE_UNRESOLVED,
    FLAG_LLM_REVIEWED,
    FLAG_LLM_UNAVAILABLE,
    FLAG_OVERRIDDEN_BY_POLICY,
    FLAG_STRICTER_THAN_POLICY,
    LLMApprover,
)
from invoice_processor.database import init_db
from invoice_processor.ingestion import extract_invoice
from invoice_processor.inventory import SQLiteInventory
from invoice_processor.llm import LLMError
from invoice_processor.models import (
    ApprovalDecision,
    Invoice,
    InvoiceItem,
    IssueCode,
    Severity,
    ValidationIssue,
    ValidationResult,
)
from invoice_processor.validation import validate_invoice

APPROVED, REJECTED = ApprovalDecision.APPROVED, ApprovalDecision.REJECTED
INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"


class ScriptedLLM:
    """Returns scripted outputs per schema, in order; records every call."""

    def __init__(self, drafts: list[dict], critiques: list[dict] | None = None, fail: bool = False) -> None:
        self.outputs = {"ApprovalDraft": list(drafts), "Critique": list(critiques or [])}
        self.fail = fail
        self.calls: list[tuple[str, str, str]] = []  # (schema, system, prompt)

    def complete_structured(self, system, prompt, schema, *, agent=None):
        self.calls.append((schema.__name__, system, prompt))
        if self.fail:
            raise LLMError("xAI request failed (UNAVAILABLE): offline")
        queue = self.outputs[schema.__name__]
        data = queue.pop(0) if len(queue) > 1 else (queue[0] if queue else {})
        return schema.model_validate(data)

    def run_tools(self, *args, **kwargs):
        raise NotImplementedError

    def schemas(self) -> list[str]:
        return [c[0] for c in self.calls]


def _draft(decision: str, scrutiny: bool, reasoning: str) -> dict:
    return {"decision": decision, "requires_additional_scrutiny": scrutiny, "reasoning": reasoning}


def _invoice(total: str) -> Invoice:
    return Invoice(invoice_number="INV-T1", vendor_name="Widgets Inc.", total=Decimal(total),
                   items=[InvoiceItem(name="WidgetA", quantity=1, unit_price=Decimal(total))])


def _validation(*issues: ValidationIssue) -> ValidationResult:
    return ValidationResult(invoice_number="INV-T1", issues=list(issues))


SUSPICIOUS = ValidationIssue(code=IssueCode.SUSPICIOUS_CONTENT, severity=Severity.WARNING, field="notes",
                             message="Invoice uses pressure language (urgent, wire transfer).")

GOOD_REASONING = ("Total ${amount} is {relation} the $10,000.00 threshold, so additional scrutiny is {req}. "
                  "Validation found no errors and no warnings. Approve.")


# --------------------------------------------------------------------------- #
# Critique corrects incorrect initial decisions
# --------------------------------------------------------------------------- #


def test_critique_corrects_approval_that_violates_policy() -> None:
    llm = ScriptedLLM(drafts=[
        _draft("approved", False, "Amount of $4,000.00 is well under $10,000 and all items are in stock. Approve."),
        _draft("rejected", False, "Total $4,000.00 is under the $10,000.00 threshold, but the notes contain "
                                  "pressure language (urgent, wire transfer), a suspicious content warning that "
                                  "blocks approval at any amount. Reject for manual review."),
    ])
    result = approve_invoice(_invoice("4000.00"), _validation(SUSPICIOUS), approver=LLMApprover(llm))

    assert result.decision is REJECTED
    assert FLAG_OVERRIDDEN_BY_POLICY not in result.flags  # the loop fixed it; no override needed
    assert FLAG_CRITIQUE_UNRESOLVED not in result.flags
    assert "suspicious content" in result.reasoning
    assert llm.schemas() == ["ApprovalDraft", "Critique", "ApprovalDraft", "Critique"]
    assert [step.split(":")[0] for step in result.review_trail] == ["Draft 1", "Critique 1", "Draft 2", "Critique 2"]
    assert "[policy_violation]" in result.review_trail[1]
    assert "[missed_validation_issue]" in result.review_trail[1]
    assert result.review_trail[3] == "Critique 2: no issues found."
    # The revision prompt carried the critique back to the agent.
    assert "policy_violation" in llm.calls[2][2]


def test_critique_corrects_misread_scrutiny_rule_at_exactly_10000() -> None:
    llm = ScriptedLLM(drafts=[
        _draft("approved", True, "Total $10,000.00 meets the $10,000 threshold so it needs additional scrutiny. "
                                 "Validation passed with no issues. Approve."),
        _draft("approved", False, GOOD_REASONING.format(amount="10,000.00", relation="not greater than", req="not required")),
    ])
    result = approve_invoice(_invoice("10000.00"), _validation(), approver=LLMApprover(llm))

    assert result.decision is APPROVED
    assert result.requires_additional_scrutiny is False
    assert "[scrutiny_rule]" in result.review_trail[1]
    assert "not greater than $10,000.00" in result.review_trail[1]
    assert result.review_trail[-1] == "Critique 2: no issues found."


def test_critique_corrects_missed_scrutiny_above_threshold() -> None:
    llm = ScriptedLLM(drafts=[
        _draft("approved", False, "Total $15,000.00; validation passed with no issues. Routine approval."),
        _draft("approved", True, GOOD_REASONING.format(amount="15,000.00", relation="greater than", req="required")),
    ])
    result = approve_invoice(_invoice("15000.00"), _validation(), approver=LLMApprover(llm))

    assert result.decision is APPROVED
    assert result.requires_additional_scrutiny is True
    assert "[scrutiny_rule]" in result.review_trail[1]
    assert "strictly greater than $10,000.00" in result.review_trail[1]


def test_critique_flags_missed_validation_issues_on_real_invoice(tmp_path: Path) -> None:
    invoice = extract_invoice(INVOICES / "invoice_1012.txt")  # OCR corrections -> 2 warnings
    validation = validate_invoice(invoice, SQLiteInventory(init_db(tmp_path / "inventory.db")))
    llm = ScriptedLLM(drafts=[
        _draft("approved", False, "Total $9,975.00 is under the $10,000.00 threshold, all items are within stock. Approve."),
        _draft("approved", False, "Total $9,975.00 is under the $10,000.00 threshold, so no additional scrutiny. "
                                  "Two extraction_uncertain warnings: the invoice date and the items[1].line_total were "
                                  "OCR-corrected; both corrected values are consistent with line math. Approve."),
    ])
    result = approve_invoice(invoice, validation, approver=LLMApprover(llm))

    assert result.decision is APPROVED
    assert "[missed_validation_issue]" in result.review_trail[1]
    assert "OCR" in result.review_trail[1]
    assert result.review_trail[-1] == "Critique 2: no issues found."


def test_llm_critic_findings_also_trigger_revision() -> None:
    first = GOOD_REASONING.format(amount="5,000.00", relation="not greater than", req="not required")
    llm = ScriptedLLM(
        drafts=[_draft("approved", False, first),
                _draft("rejected", False, first.replace("Approve.", "However the vendor address does not match prior "
                                                       "invoices from Widgets Inc.; reject for manual verification."))],
        critiques=[{"findings": [{"category": "insufficient_justification",
                                  "detail": "Vendor address differs from previous invoices; approval does not address it."}]},
                   {"findings": []}],
    )
    result = approve_invoice(_invoice("5000.00"), _validation(), approver=LLMApprover(llm))

    assert result.decision is REJECTED
    assert FLAG_STRICTER_THAN_POLICY in result.flags  # agent may be stricter than the rules
    assert "Vendor address differs" in result.review_trail[1]


# --------------------------------------------------------------------------- #
# Loop bounds and deterministic constraints
# --------------------------------------------------------------------------- #


def test_correct_first_draft_needs_only_one_critique() -> None:
    llm = ScriptedLLM(drafts=[_draft("approved", False, GOOD_REASONING.format(
        amount="5,000.00", relation="not greater than", req="not required"))])
    result = approve_invoice(_invoice("5000.00"), _validation(), approver=LLMApprover(llm))

    assert result.decision is APPROVED
    assert llm.schemas() == ["ApprovalDraft", "Critique"]
    assert result.flags == [FLAG_LLM_REVIEWED]
    assert result.reviewer == "llm-approval-agent"


def test_stubborn_agent_is_bounded_and_overridden_by_policy() -> None:
    stubborn = _draft("approved", False, "Total $4,000.00 is fine. Approve.")
    llm = ScriptedLLM(drafts=[stubborn])  # repeats the same wrong draft forever
    result = approve_invoice(_invoice("4000.00"), _validation(SUSPICIOUS), approver=LLMApprover(llm))

    assert llm.schemas() == ["ApprovalDraft", "Critique", "ApprovalDraft", "Critique", "ApprovalDraft"]  # max 5 calls
    assert result.decision is REJECTED
    assert {FLAG_OVERRIDDEN_BY_POLICY, FLAG_CRITIQUE_UNRESOLVED, FLAG_BLOCKING_WARNINGS} <= set(result.flags)
    assert "constrained to REJECTED by deterministic policy" in result.reasoning


def test_scrutiny_always_comes_from_policy() -> None:
    llm = ScriptedLLM(drafts=[_draft("approved", False, "Total $50,000.00 is routine. Validation passed. Approve.")])
    result = approve_invoice(_invoice("50000.00"), _validation(), approver=LLMApprover(llm))
    assert result.requires_additional_scrutiny is True
    assert "Scrutiny set by policy" in result.reasoning


def test_hard_failures_cannot_be_approved() -> None:
    error = ValidationIssue(code=IssueCode.INSUFFICIENT_STOCK, severity=Severity.ERROR, item="GadgetX",
                            message="Invoice bills 20 units of GadgetX, but only 5 are in stock.")
    llm = ScriptedLLM(drafts=[_draft("approved", False, "Stock will be replenished soon; GadgetX shortfall is fine.")])
    result = approve_invoice(_invoice("5000.00"), _validation(error), approver=LLMApprover(llm))
    assert result.decision is REJECTED


@pytest.mark.parametrize("rounds", [-1, 3])
def test_critique_rounds_are_capped(rounds: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 2"):
        LLMApprover(ScriptedLLM(drafts=[]), critique_rounds=rounds)


def test_zero_rounds_skips_critique_but_keeps_constraints() -> None:
    llm = ScriptedLLM(drafts=[_draft("approved", False, "Fine. Approve.")])
    result = approve_invoice(_invoice("4000.00"), _validation(SUSPICIOUS), approver=LLMApprover(llm, critique_rounds=0))
    assert llm.schemas() == ["ApprovalDraft"]
    assert result.decision is REJECTED


def test_llm_outage_falls_back_to_deterministic_policy() -> None:
    result = approve_invoice(_invoice("5000.00"), _validation(), approver=LLMApprover(ScriptedLLM([], fail=True)))
    assert result.decision is APPROVED
    assert FLAG_LLM_UNAVAILABLE in result.flags
    assert "Deterministic policy decision used" in result.reasoning


def test_prompts_give_the_critic_policy_facts() -> None:
    llm = ScriptedLLM(drafts=[_draft("approved", False, GOOD_REASONING.format(
        amount="5,000.00", relation="not greater than", req="not required"))])
    approve_invoice(_invoice("5000.00"), _validation(), approver=LLMApprover(llm))
    draft_prompt, critic_prompt = llm.calls[0][2], llm.calls[1][2]

    assert "strictly greater than $10,000.00 USD" in draft_prompt
    assert "POLICY ENGINE FACTS" not in draft_prompt  # the agent reasons independently
    assert "POLICY ENGINE FACTS" in critic_prompt and "Policy engine decision: APPROVED" in critic_prompt
