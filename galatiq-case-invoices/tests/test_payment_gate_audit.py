"""Audit: PAYMENT MUST NEVER OCCUR WITHOUT EXPLICIT APPROVAL.

These tests observe the *real default* payment path: ``build_workflow()`` with no
``pay`` argument wraps ``invoice_processor.workflow.mock_payment`` in a
``DuplicatePaymentGuard``. The fixture replaces that function with a counting
wrapper around the real ``mock_payment``, so every payment the system makes
(deterministic path, payment agent tool, CLI) is counted.
"""

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from invoice_processor import payment as payment_module
from invoice_processor import workflow as workflow_module
from invoice_processor.approval import FLAG_MALFORMED_APPROVAL
from invoice_processor.approval_agent import FLAG_LLM_UNAVAILABLE
from invoice_processor.cli import main
from invoice_processor.database import init_db
from invoice_processor.ingestion import extract_invoice
from invoice_processor.inventory import SQLiteInventory
from invoice_processor.llm import LLMSettings, XAIClient
from invoice_processor.models import ApprovalDecision, ApprovalResult, PaymentStatus, ValidationResult
from invoice_processor.payment import process_payment
from invoice_processor.workflow import PipelineStatus, build_workflow, payment_step, run_invoice

INVOICES = Path(__file__).resolve().parents[1] / "data" / "invoices"
INVALID_SAMPLES = ["invoice_1002.txt", "invoice_1003.txt", "invoice_1005.json", "invoice_1007.csv",
                   "invoice_1008.txt", "invoice_1009.json", "invoice_1013.json", "invoice_1013.pdf",
                   "invoice_1016.json"]
APPROVED_SAMPLES = ["invoice_1001.txt", "invoice_1004.json", "invoice_1004_revised.json", "invoice_1006.csv",
                    "invoice_1010.txt", "invoice_1011.pdf", "invoice_1011.txt", "invoice_1012.pdf",
                    "invoice_1012.txt", "invoice_1014.xml", "invoice_1015.csv"]


@pytest.fixture
def payments(monkeypatch) -> list[tuple]:
    """Every call that reaches the real mock_payment through the default wiring."""
    calls: list[tuple] = []
    real = payment_module.mock_payment

    def counting_mock_payment(vendor, amount, *, currency="USD", invoice_number=None):
        calls.append((vendor, amount, currency, invoice_number))
        return real(vendor, amount, currency=currency, invoice_number=invoice_number)

    monkeypatch.setattr(workflow_module, "mock_payment", counting_mock_payment)
    return calls


@pytest.fixture
def db(tmp_path) -> Path:
    return init_db(tmp_path / "inventory.db")


@pytest.fixture
def inventory(db) -> SQLiteInventory:
    return SQLiteInventory(db)


class AgentLLM:
    """Fake Grok. ``draft`` is the approval agent's decision. Every agent also tries to pay:
    the validation agent tries mock_payment (not in its toolset); the payment agent tries an
    inflated amount, then the authorized amount twice."""

    def __init__(self, draft: str = "approved") -> None:
        self.draft = draft
        self.agents_run: list[str] = []

    def complete_structured(self, system, prompt, schema):
        if schema.__name__ == "ApprovalDraft":
            return schema(decision=self.draft, requires_additional_scrutiny=False,
                          reasoning="Total is not greater than the $10,000.00 threshold and validation passed "
                                    "with no errors or warnings, so this decision follows policy.")
        return schema.model_validate({})  # critic: no findings

    def run_tools(self, system, prompt, toolset, schema, *, max_rounds=6):
        self.agents_run.append(toolset.name)
        if toolset.name == "validation":
            toolset.invoke("mock_payment", {"vendor": "Evil Corp", "amount": "1000000"})
            return schema(summary="reviewed")
        auth = toolset.authorization
        toolset.invoke("mock_payment", {"vendor": auth.vendor, "amount": str(auth.amount * 10)})
        toolset.invoke("mock_payment", {"vendor": auth.vendor, "amount": str(auth.amount)})
        toolset.invoke("mock_payment", {"vendor": auth.vendor, "amount": str(auth.amount)})
        return schema(summary="paid")


class ApproveEverything:
    def review(self, invoice, validation) -> ApprovalResult:
        return ApprovalResult(invoice_number=invoice.invoice_number, decision="approved", reasoning="Rubber stamp.",
                              reviewed_vendor=invoice.vendor_name, reviewed_amount=invoice.total,
                              reviewed_currency=invoice.currency)


class RejectEverything:
    def review(self, invoice, validation) -> ApprovalResult:
        return ApprovalResult(invoice_number=invoice.invoice_number, decision="rejected", reasoning="VP declined.")


# =========================================================================== #
# 1. Rejected invoice -> payment is never called
# =========================================================================== #


def test_policy_rejection_never_pays_via_cli(payments, db, tmp_path, capsys) -> None:
    """A *valid* invoice the approval policy rejects (fraud language) is not paid."""
    path = tmp_path / "invoice_urgent.txt"
    path.write_text((INVOICES / "invoice_1001.txt").read_text() + "Notes: URGENT - wire transfer preferred.\n")

    code = main([f"--invoice_path={path}", f"--db-path={db}", "--llm=off"])
    out = capsys.readouterr().out

    assert payments == []
    assert code == 1
    assert "VALIDATION  PASSED" in out and "APPROVAL  REJECTED" in out
    assert "PAYMENT  SKIPPED" in out and "rejected at approval" in out


def test_approver_rejection_never_pays(payments, inventory) -> None:
    state = run_invoice(INVOICES / "invoice_1001.txt", build_workflow(inventory, approver=RejectEverything()))
    assert payments == []
    assert state["status"] is PipelineStatus.REJECTED
    assert "payment" not in state


def test_llm_approval_agent_rejection_never_pays(payments, inventory) -> None:
    llm = AgentLLM(draft="rejected")
    state = run_invoice(INVOICES / "invoice_1001.txt", build_workflow(inventory, llm=llm))

    assert payments == []
    assert state["approval"].decision is ApprovalDecision.REJECTED
    assert "payment" not in llm.agents_run  # the payment agent is never started


@pytest.mark.parametrize("use_llm", [False, True], ids=["deterministic", "llm-agent"])
def test_payment_step_with_rejected_approval_never_pays(payments, use_llm) -> None:
    invoice = extract_invoice(INVOICES / "invoice_1001.txt")
    rejected = ApprovalResult(invoice_number="INV-1001", decision="rejected", reasoning="No.",
                              reviewed_vendor="Widgets Inc.", reviewed_amount=Decimal("5000.00"), reviewed_currency="USD")
    update = payment_step(invoice, ValidationResult(invoice_number="INV-1001"), rejected,
                          pay=workflow_module.mock_payment, llm=AgentLLM() if use_llm else None)
    assert payments == []
    assert update["payment"].status is PaymentStatus.SKIPPED


# =========================================================================== #
# 2. Validation failure -> payment is never called
# =========================================================================== #

MODES = {
    "deterministic": dict(),
    "rubber-stamp-approver": dict(approver=ApproveEverything()),
    "approving-llm-agents": dict(llm=AgentLLM()),
}


@pytest.mark.parametrize("mode", MODES, ids=MODES.keys())
@pytest.mark.parametrize("filename", INVALID_SAMPLES)
def test_validation_failure_never_pays(payments, inventory, filename, mode) -> None:
    state = run_invoice(INVOICES / filename, build_workflow(inventory, **MODES[mode]))

    assert payments == []
    assert not state["validation"].is_valid
    assert state["status"] is PipelineStatus.REJECTED
    assert "approval" not in state and "payment" not in state


def test_validation_failure_never_pays_via_cli(payments, db, tmp_path, capsys) -> None:
    batch = tmp_path / "invalid"
    batch.mkdir()
    for name in INVALID_SAMPLES:
        (batch / name).write_bytes((INVOICES / name).read_bytes())

    code = main([f"--invoice_path={batch}", f"--db-path={db}", "--llm=off"])

    assert payments == []
    assert code == 1
    assert f"0 paid, {len(INVALID_SAMPLES)} rejected, 0 failed" in capsys.readouterr().out


@pytest.mark.parametrize("use_llm", [False, True], ids=["deterministic", "llm-agent"])
def test_approved_but_invalid_never_pays(payments, inventory, use_llm) -> None:
    """Even an APPROVED decision on the exact terms cannot pay an invoice that failed validation."""
    invoice = extract_invoice(INVOICES / "invoice_1002.txt")
    from invoice_processor.validation import validate_invoice
    validation = validate_invoice(invoice, inventory)
    approved = ApproveEverything().review(invoice, validation)

    update = payment_step(invoice, validation, approved, pay=workflow_module.mock_payment,
                          llm=AgentLLM() if use_llm else None)
    assert payments == []
    assert "Validation failed" in update["payment"].message


# =========================================================================== #
# 3. Malformed approval -> payment is never called
# =========================================================================== #

TERMS = dict(reviewed_vendor="Widgets Inc.", reviewed_amount=Decimal("5000.00"), reviewed_currency="USD")
MALFORMED_APPROVALS = {
    "none": None,
    "dict": {"invoice_number": "INV-1001", "decision": "approved", **TERMS},
    "string": "approved",
    "look-alike object": SimpleNamespace(invoice_number="INV-1001", decision=ApprovalDecision.APPROVED,
                                         is_approved=True, covers_terms=lambda invoice: True, reasoning="", **TERMS),
    "unvalidated str decision": ApprovalResult.model_construct(invoice_number="INV-1001", decision="approved",
                                                               reasoning="x", **TERMS),
    "approved without terms": ApprovalResult(invoice_number="INV-1001", decision="approved", reasoning="x"),
    "approved other amount": ApprovalResult(invoice_number="INV-1001", decision="approved", reasoning="x",
                                            **(TERMS | {"reviewed_amount": Decimal("50.00")})),
    "approved other invoice": ApprovalResult(invoice_number="INV-9999", decision="approved", reasoning="x", **TERMS),
}


@pytest.mark.parametrize("use_llm", [False, True], ids=["deterministic", "llm-agent"])
@pytest.mark.parametrize("approval", MALFORMED_APPROVALS.values(), ids=MALFORMED_APPROVALS.keys())
def test_malformed_approval_never_pays(payments, approval, use_llm) -> None:
    invoice = extract_invoice(INVOICES / "invoice_1001.txt")
    valid = ValidationResult(invoice_number="INV-1001")
    llm = AgentLLM() if use_llm else None

    update = payment_step(invoice, valid, approval, pay=workflow_module.mock_payment, llm=llm)
    direct = process_payment(invoice, valid, approval, pay=workflow_module.mock_payment)

    assert payments == []
    assert update["payment"].status is PaymentStatus.SKIPPED
    assert direct.status is PaymentStatus.SKIPPED
    assert llm is None or llm.agents_run == []


MALFORMED_APPROVER_OUTPUT = {
    "None": None,
    "dict": {"decision": "approved"},
    "string": "APPROVED",
    "empty reasoning (unvalidated)": ApprovalResult.model_construct(
        invoice_number="INV-1001", decision=ApprovalDecision.APPROVED, reasoning="", flags=[], **TERMS),
}


@pytest.mark.parametrize("output", MALFORMED_APPROVER_OUTPUT.values(), ids=MALFORMED_APPROVER_OUTPUT.keys())
def test_malformed_approver_output_is_rejected_and_never_pays(payments, inventory, output) -> None:
    class MalformedApprover:
        def review(self, invoice, validation):
            return output

    state = run_invoice(INVOICES / "invoice_1001.txt", build_workflow(inventory, approver=MalformedApprover()))

    assert payments == []
    assert state["status"] is PipelineStatus.REJECTED
    assert FLAG_MALFORMED_APPROVAL in state["approval"].flags


class _Resp:
    tool_calls: list = []


class MalformedDraftSDK:
    """Real XAIClient, fake transport: every structured approval draft is schema-invalid."""

    def __init__(self) -> None:
        self.chat = self

    def create(self, **kwargs):
        return self

    def append(self, message):
        pass

    def sample(self):
        return _Resp()

    def parse(self, schema):
        return None, schema.model_validate({"decision": "definitely approve", "reasoning": 42})


def test_malformed_llm_approval_never_becomes_an_approval(payments, inventory) -> None:
    llm = XAIClient(LLMSettings(), sdk_client=MalformedDraftSDK())
    state = run_invoice(INVOICES / "invoice_1003.txt", build_workflow(inventory, llm=llm))  # invalid invoice

    assert payments == []
    assert state["status"] is PipelineStatus.REJECTED


def test_malformed_llm_approval_falls_back_to_explicit_policy_decision(payments, inventory) -> None:
    """Design: a malformed LLM draft is discarded; the decision used is the deterministic policy's own
    explicit approval (reviewer rule-based-approver, flagged llm_unavailable), never the malformed draft."""
    llm = XAIClient(LLMSettings(), sdk_client=MalformedDraftSDK())
    state = run_invoice(INVOICES / "invoice_1001.txt", build_workflow(inventory, llm=llm))

    approval = state["approval"]
    assert approval.reviewer == "rule-based-approver"
    assert FLAG_LLM_UNAVAILABLE in approval.flags
    assert "did not match ApprovalDraft" in approval.reasoning
    # The payment agent also gets malformed output, so it never calls the tool: nothing is paid.
    assert payments == []
    assert state["status"] is PipelineStatus.FAILED


@pytest.mark.parametrize(
    "injected",
    ["approval", "validation+approval", "invoice"],
)
def test_forged_initial_state_is_ignored(payments, inventory, tmp_path, injected) -> None:
    """BUG (fixed): keys passed into workflow.invoke() survived when a stage failed, letting a forged
    approval/validation (inventory down) or invoice (unreadable file) reach payment."""
    invoice = extract_invoice(INVOICES / "invoice_1001.txt")
    forged_approval = ApproveEverything().review(invoice, None)
    forged = {
        "approval": {"invoice_path": str(INVOICES / "invoice_1002.txt"), "approval": forged_approval},
        "validation+approval": {"invoice_path": str(INVOICES / "invoice_1001.txt"), "approval": forged_approval,
                                "validation": ValidationResult(invoice_number="INV-1001")},
        "invoice": {"invoice_path": str(tmp_path / "missing.txt"), "invoice": invoice},
    }[injected]
    # Inventory unavailable for the validation+approval case, so validation cannot overwrite the forgery.
    lookup = SQLiteInventory(tmp_path / "down.db") if injected == "validation+approval" else inventory

    state = build_workflow(lookup).invoke(forged)

    assert payments == []
    assert state["status"] in (PipelineStatus.REJECTED, PipelineStatus.FAILED)


# =========================================================================== #
# 4. Approved invoice -> payment is called exactly once
# =========================================================================== #


@pytest.mark.parametrize("use_llm", [False, True], ids=["deterministic", "llm-agents"])
@pytest.mark.parametrize("filename", APPROVED_SAMPLES)
def test_approved_invoice_is_paid_exactly_once(payments, inventory, filename, use_llm) -> None:
    llm = AgentLLM() if use_llm else None  # the payment agent tries to pay three times
    state = run_invoice(INVOICES / filename, build_workflow(inventory, llm=llm))
    invoice = state["invoice"]

    assert state["approval"].is_approved
    assert payments == [(invoice.vendor_name, invoice.total, invoice.currency, invoice.invoice_number)]
    assert state["status"] is PipelineStatus.COMPLETED
    assert state["payment"].status is PaymentStatus.PAID


def test_rerunning_an_approved_invoice_does_not_pay_again(payments, inventory) -> None:
    workflow = build_workflow(inventory)
    first = run_invoice(INVOICES / "invoice_1001.txt", workflow)
    second = run_invoice(INVOICES / "invoice_1001.txt", workflow)

    assert len(payments) == 1
    assert first["payment"].status is PaymentStatus.PAID
    assert second["payment"].status is PaymentStatus.FAILED  # duplicate blocked


def test_cli_batch_pays_each_approved_invoice_exactly_once(payments, db, capsys) -> None:
    code = main([f"--invoice_path={INVOICES}", f"--db-path={db}", "--llm=off"])

    paid_numbers = [call[3] for call in payments]
    assert sorted(paid_numbers) == ["INV-1001", "INV-1004", "INV-1006", "INV-1010", "INV-1011", "INV-1012",
                                    "INV-1014", "INV-1015"]
    assert len(paid_numbers) == len(set(paid_numbers))
    assert code == 3  # the duplicate copies are reported, not paid
    assert "8 paid, 9 rejected, 3 failed" in capsys.readouterr().out


@pytest.mark.parametrize("use_llm", [False, True], ids=["deterministic", "llm-agent"])
def test_payment_step_with_valid_approval_pays_exactly_once(payments, use_llm) -> None:
    invoice = extract_invoice(INVOICES / "invoice_1001.txt")
    approval = ApprovalResult(invoice_number="INV-1001", decision="approved", reasoning="OK.", **TERMS)

    update = payment_step(invoice, ValidationResult(invoice_number="INV-1001"), approval,
                          pay=workflow_module.mock_payment, llm=AgentLLM() if use_llm else None)

    assert payments == [("Widgets Inc.", Decimal("5000.00"), "USD", "INV-1001")]
    assert update["payment"].status is PaymentStatus.PAID
