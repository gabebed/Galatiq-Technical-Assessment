"""Proofs that no agent can cause a payment without an APPROVED approval result.

``AdversarialLLM`` stands in for Grok and, in every agent, tries to pay an
attacker, pay an inflated amount, call tools it does not have, and inject SQL.
"""

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from invoice_processor.agent_tools import PaymentToolset
from invoice_processor.database import init_db
from invoice_processor.inventory import SQLiteInventory
from invoice_processor.llm import LLMError
from invoice_processor.models import (
    ApprovalDecision,
    ApprovalResult,
    Invoice,
    PaymentResult,
    PaymentStatus,
    ValidationResult,
)
from invoice_processor.payment import mock_payment
from invoice_processor.tools import Toolset, ToolResult
from invoice_processor.workflow import PipelineStatus, build_workflow, payment_step, run_invoice

INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"
APPROVED_SAMPLES = {
    "invoice_1001.txt", "invoice_1004.json", "invoice_1004_revised.json", "invoice_1006.csv",
    "invoice_1010.txt", "invoice_1011.txt", "invoice_1011.pdf", "invoice_1012.txt", "invoice_1012.pdf",
    "invoice_1014.xml", "invoice_1015.csv",
}

Script = list[tuple[str, dict]] | Callable[[Toolset], list[tuple[str, dict]]]

ATTACKS: list[tuple[str, dict]] = [
    ("mock_payment", {"vendor": "Evil Corp", "amount": "1000000.00"}),
    ("execute_sql", {"sql": "UPDATE inventory SET stock = 999"}),
    ("python", {"code": "import os; os.system('echo pwned')"}),
    ("lookup_inventory", {"item": "WidgetA'; DROP TABLE inventory; --"}),
]


def _payment_attacks(toolset: Toolset) -> list[tuple[str, dict]]:
    """Inflated amount, then the honest call, then a duplicate (fits the 3-call budget)."""
    auth = toolset.authorization  # type: ignore[attr-defined]
    honest = {"vendor": auth.vendor, "amount": str(auth.amount)}
    return [
        ("mock_payment", {"vendor": auth.vendor, "amount": str(auth.amount * 10)}),
        ("mock_payment", honest),
        ("mock_payment", honest),
    ]


def _relentless_attacks(toolset: Toolset) -> list[tuple[str, dict]]:
    """Attacks first; by the time the agent tries the honest call its budget is spent."""
    auth = toolset.authorization  # type: ignore[attr-defined]
    return [*ATTACKS, ("mock_payment", {"vendor": auth.vendor, "amount": str(auth.amount)})]


class AdversarialLLM:
    """Scripted stand-in for an LLM agent. Calls tools only via Toolset.invoke, like XAIClient."""

    def __init__(self, scripts: dict[str, Script] | None = None, fail: bool = False) -> None:
        self.scripts = scripts if scripts is not None else {"validation": ATTACKS, "payment": _payment_attacks}
        self.fail = fail
        self.agents_run: list[str] = []
        self.results: list[tuple[str, ToolResult]] = []

    def complete_structured(self, system, prompt, schema, *, agent=None):
        """Approval agent: always approve, never scrutinize. Critic: never object."""
        self.agents_run.append(f"approval:{schema.__name__}")
        if self.fail:
            raise LLMError("model unavailable")
        if schema.__name__ == "ApprovalDraft":
            return schema(decision="approved", requires_additional_scrutiny=False, reasoning="Pay it.")
        return schema.model_validate({})

    def run_tools(self, system, prompt, toolset, schema, *, max_rounds=6, agent=None):
        self.agents_run.append(toolset.name)
        if self.fail:
            raise LLMError("model unavailable")
        script = self.scripts.get(toolset.name, [])
        for name, args in script(toolset) if callable(script) else script:
            self.results.append((toolset.name, toolset.invoke(name, args)))
        return schema.model_validate({"summary": f"{toolset.name} agent finished."})


class SpyPayment:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, vendor, amount, *, currency="USD", invoice_number=None) -> PaymentResult:
        self.calls.append((vendor, amount, currency, invoice_number))
        return mock_payment(vendor, amount, currency=currency, invoice_number=invoice_number)


@pytest.fixture
def inventory(tmp_path: Path) -> SQLiteInventory:
    return SQLiteInventory(init_db(tmp_path / "inventory.db"))


# --------------------------------------------------------------------------- #
# The payment step itself: requires approval.decision == APPROVED
# --------------------------------------------------------------------------- #


def _invoice() -> Invoice:
    from invoice_processor.ingestion import extract_invoice
    return extract_invoice(INVOICES / "invoice_1001.txt")


NOT_APPROVED = {
    "no approval": None,
    "rejected": ApprovalResult(invoice_number="INV-1001", decision=ApprovalDecision.REJECTED, reasoning="No."),
    "approved for another invoice": ApprovalResult(invoice_number="INV-1002", decision=ApprovalDecision.APPROVED,
                                                   reasoning="Approved."),
}


@pytest.mark.parametrize("use_llm", [False, True], ids=["deterministic", "llm-agent"])
@pytest.mark.parametrize("approval", NOT_APPROVED.values(), ids=NOT_APPROVED.keys())
def test_payment_step_never_pays_without_approved_result(approval, use_llm: bool) -> None:
    spy, llm = SpyPayment(), AdversarialLLM()
    update = payment_step(_invoice(), ValidationResult(invoice_number="INV-1001"), approval,
                          pay=spy, llm=llm if use_llm else None)

    assert spy.calls == []
    assert update["payment"].status is PaymentStatus.SKIPPED
    assert update["status"] is PipelineStatus.REJECTED
    assert "payment" not in llm.agents_run  # the payment agent is never even started


@pytest.mark.parametrize("use_llm", [False, True], ids=["deterministic", "llm-agent"])
def test_payment_step_never_pays_approved_but_invalid_invoice(use_llm: bool, inventory: SQLiteInventory) -> None:
    from invoice_processor.ingestion import extract_invoice
    from invoice_processor.validation import validate_invoice

    invoice = extract_invoice(INVOICES / "invoice_1002.txt")  # 20x GadgetX, 5 in stock
    validation = validate_invoice(invoice, inventory)
    forged = ApprovalResult(invoice_number="INV-1002", decision=ApprovalDecision.APPROVED, reasoning="Forged.")
    spy = SpyPayment()

    update = payment_step(invoice, validation, forged, pay=spy, llm=AdversarialLLM() if use_llm else None)

    assert spy.calls == []
    assert "Validation failed" in update["payment"].message


APPROVED_1001 = ApprovalResult(invoice_number="INV-1001", decision=ApprovalDecision.APPROVED, reasoning="OK.",
                               reviewed_vendor="Widgets Inc.", reviewed_amount=Decimal("5000.00"),
                               reviewed_currency="USD")


def test_payment_step_pays_only_the_authorized_payment_despite_attacks() -> None:
    spy, llm = SpyPayment(), AdversarialLLM()
    approval = APPROVED_1001
    update = payment_step(_invoice(), ValidationResult(invoice_number="INV-1001"), approval, pay=spy, llm=llm)

    assert spy.calls == [("Widgets Inc.", Decimal("5000.00"), "USD", "INV-1001")]
    assert update["status"] is PipelineStatus.COMPLETED
    payment_results = [r for agent, r in llm.results if agent == "payment"]
    assert [r.ok for r in payment_results] == [False, True, False]  # inflated, honest, duplicate
    assert "does not match the approved payment" in payment_results[0].error
    assert "already been paid" in payment_results[2].error
    assert len(update["tool_calls"]) == 3


def test_misbehaving_payment_agent_exhausts_its_budget_and_pays_nothing() -> None:
    spy, llm = SpyPayment(), AdversarialLLM(scripts={"payment": _relentless_attacks})
    approval = APPROVED_1001
    update = payment_step(_invoice(), ValidationResult(invoice_number="INV-1001"), approval, pay=spy, llm=llm)

    assert spy.calls == []
    assert update["status"] is PipelineStatus.FAILED  # left for a human, never silently paid
    assert "budget" in llm.results[-1][1].error


# --------------------------------------------------------------------------- #
# Whole workflow with adversarial agents
# --------------------------------------------------------------------------- #


def test_adversarial_agents_only_pay_approved_invoices(inventory: SQLiteInventory) -> None:
    spy, llm = SpyPayment(), AdversarialLLM()
    workflow = build_workflow(inventory, pay=spy, llm=llm)

    paid = {}
    for path in sorted(INVOICES.iterdir()):
        before = len(spy.calls)
        state = run_invoice(path, workflow)
        approved = state.get("approval") is not None and state["approval"].is_approved
        new_calls = spy.calls[before:]

        if approved:
            invoice = state["invoice"]
            assert new_calls == [(invoice.vendor_name, invoice.total, invoice.currency, invoice.invoice_number)]
            paid[path.name] = state["status"]
        else:
            assert new_calls == [], f"{path.name} reached the payment function without approval"
            assert "payment" not in state

    assert set(paid) == APPROVED_SAMPLES
    assert all(status is PipelineStatus.COMPLETED for status in paid.values())
    assert not any(call[0] == "Evil Corp" for call in spy.calls)


def test_approval_rejection_blocks_adversarial_payment_agent(inventory: SQLiteInventory) -> None:
    class RejectAll:
        def review(self, invoice: Invoice, validation: ValidationResult) -> ApprovalResult:
            return ApprovalResult(invoice_number=invoice.invoice_number, decision=ApprovalDecision.REJECTED,
                                  reasoning="VP declined.")

    spy, llm = SpyPayment(), AdversarialLLM()
    state = run_invoice(INVOICES / "invoice_1001.txt",
                        build_workflow(inventory, pay=spy, llm=llm, approver=RejectAll()))

    assert state["status"] is PipelineStatus.REJECTED
    assert spy.calls == []
    assert llm.agents_run == ["validation"]  # explicit approver replaces the LLM approver; no payment agent


def test_validation_agent_cannot_reach_payment_or_modify_inventory(inventory: SQLiteInventory) -> None:
    llm = AdversarialLLM(scripts={"validation": ATTACKS})
    state = run_invoice(INVOICES / "invoice_1001.txt", build_workflow(inventory, pay=SpyPayment(), llm=llm))

    calls = [c for c in state["tool_calls"] if c.toolset == "validation"]
    assert [(c.tool, c.ok) for c in calls] == [
        ("mock_payment", False), ("execute_sql", False), ("python", False), ("lookup_inventory", True)]
    assert inventory.get_stock_levels(["WidgetA", "WidgetB", "GadgetX", "FakeItem"]) == {
        "WidgetA": 15, "WidgetB": 10, "GadgetX": 5, "FakeItem": 0}


def test_validation_agent_is_advisory_only(inventory: SQLiteInventory) -> None:
    path = INVOICES / "invoice_1001.txt"
    plain = run_invoice(path, build_workflow(inventory))
    with_agent = run_invoice(path, build_workflow(inventory, llm=AdversarialLLM(scripts={})))

    assert with_agent["validation"].issues == plain["validation"].issues
    assert with_agent["validation_review"].summary == "validation agent finished."
    assert with_agent["approval"].decision is plain["approval"].decision


def test_payment_agent_that_does_not_pay_leaves_invoice_unpaid(inventory: SQLiteInventory) -> None:
    spy = SpyPayment()
    llm = AdversarialLLM(scripts={"validation": [], "payment": []})
    state = run_invoice(INVOICES / "invoice_1001.txt", build_workflow(inventory, pay=spy, llm=llm))

    assert spy.calls == []
    assert state["status"] is PipelineStatus.FAILED
    assert state["payment"].status is PaymentStatus.FAILED
    assert "did not execute the payment" in state["error"]


def test_llm_outage_fails_safe(inventory: SQLiteInventory) -> None:
    spy = SpyPayment()
    state = run_invoice(INVOICES / "invoice_1001.txt",
                        build_workflow(inventory, pay=spy, llm=AdversarialLLM(fail=True)))

    assert spy.calls == []
    assert state["status"] is PipelineStatus.FAILED
    assert "model unavailable" in state["error"]
    assert state["agent_errors"] == ["validation agent: model unavailable"]
    assert state["validation"].is_valid  # deterministic validation unaffected


def test_payment_toolset_is_bound_to_its_authorization() -> None:
    assert issubclass(PaymentToolset, Toolset)
