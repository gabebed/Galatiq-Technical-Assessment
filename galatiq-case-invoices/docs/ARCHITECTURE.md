# Architecture

This document covers the design in more depth than the [README](../README.md): the data contracts, the
workflow state, where the LLM is and is not trusted, and how each failure is handled.

## 1. Principles

1. **Typed contracts between stages.** Stages exchange immutable Pydantic models, never free text.
2. **Record, then judge.** Extraction records what a document *says*, including invalid values;
   validation decides what is acceptable. Ingestion never crashes on bad business data.
3. **Deterministic core, LLM at the edges.** Code that reads inventory or moves money is deterministic,
   tested, and constrained. LLM agents investigate and reason, but their outputs are advisory, bounded by
   policy, or bound to a single authorized action.
4. **Fail closed.** If something cannot be verified (database unavailable, corrupt stock, ambiguous
   currency, LLM failure), the invoice is not paid.
5. **Offline-capable.** Every stage has a deterministic path, so the system runs without network access.

## 2. Data contracts ([models.py](../src/invoice_processor/models.py))

```
Invoice ──► ValidationResult ──► ApprovalResult ──► PaymentResult
```

| Model | Key fields | Notes |
|---|---|---|
| `InvoiceItem` | `name`, `quantity`, `unit_price`, `line_total`, `note` | Quantity is signed so negatives can be reported. Money is `Decimal`. |
| `Invoice` | header fields (all optional), `currency`, `items`, stated `subtotal` / `tax_*` / `shipping` / `total`, `extraction_warnings`, `source_file` | `computed_subtotal` and `item_quantities()` (which combines repeated lines) are used by validation. |
| `ValidationIssue` | `code` (12 `IssueCode`s), `severity` (ERROR/WARNING), `message`, `field`, `item`, `expected`, `actual` | Every issue explains itself. |
| `ValidationResult` | `issues`, computed `is_valid` (true when there are no ERRORs) | |
| `ApprovalResult` | `decision`, `reasoning`, `requires_additional_scrutiny`, `flags`, `reviewer`, `review_trail`, `reviewed_vendor` / `reviewed_amount` / `reviewed_currency` | The `reviewed_*` fields bind an approval to the exact terms it was made on. |
| `PaymentResult` | `status` (PAID/FAILED/SKIPPED), `vendor_name`, `amount`, `currency`, `transaction_id`, `message` | The model itself refuses PAID without a vendor and a positive amount. |

All models are frozen and reject unknown fields.

## 3. Workflow ([workflow.py](../src/invoice_processor/workflow.py))

A LangGraph `StateGraph` with five nodes: `ingest`, `validate`, `approve`, `reject`, `pay`.
Conditional edges route on stage results (see the README diagram).

```python
class InvoiceState(TypedDict, total=False):
    invoice_path: str
    invoice: Invoice
    validation: ValidationResult
    validation_review: ValidationReview        # validation agent, advisory
    approval: ApprovalResult
    payment: PaymentResult
    payment_report: PaymentReport              # payment agent's own summary (not trusted)
    status: PipelineStatus                     # completed | rejected | failed
    rejection_reason: str
    error: str
    tool_calls: Annotated[list[ToolCallRecord], operator.add]   # audit trail across agents
    agent_errors: Annotated[list[str], operator.add]
    llm_calls: list[LLMCallRecord]             # added by the CLI after the run, not by a node
```

How state moves through the graph:

- **Partial updates.** Each node returns only the keys it changes, and LangGraph merges them in.
  `tool_calls` and `agent_errors` accumulate across nodes (`operator.add`).
- **Input is limited to `invoice_path`** (`WorkflowInput`). Any other key a caller passes, such as a forged
  `approval`, is dropped.
- **A failed stage ends the run.** The routing functions stop on `status == FAILED`, so they never depend
  on a key being absent.

`build_workflow(inventory, pay=None, approver=None, policy=None, llm=None)` takes its collaborators
explicitly:

- **`inventory`** is any `InventoryLookup`.
- **`pay`** defaults to `mock_payment` behind a `DuplicatePaymentGuard`.
- **`llm`** switches on the three agents.

Tests use this to substitute spies, adversarial approvers, and fake LLMs.

## 4. Ingestion ([ingestion/](../src/invoice_processor/ingestion))

```
extract_invoice(path)
  ├─ .txt → text_parser.parse_text
  ├─ .pdf → pdf.pdf_to_text (pdfplumber) → text_parser.parse_text
  ├─ .json / .csv / .xml → structured.parse_*
  └─ RawInvoice ──► builder.build_invoice (single normalizer) ──► Invoice
```

- **Text parser.**
  - It recognizes labels by vocabulary, including abbreviations (`Vndr`, `Inv #`, `Due Dt`).
  - It ranks competing labels, so an email's `From:` loses to `Vendor:`.
  - It splits several labels on one line (`Vendor: … Due: …`).
  - It reads three item layouts: `qty:` lines, `- Item x12` bullets, and tables.
- **CSV.** Supports both `field,value` and row-per-item layouts.
- **Normalizer** ([normalize.py](../src/invoice_processor/ingestion/normalize.py)):
  - money becomes `Decimal`
  - 10 date formats (slash dates are read month/day)
  - scan damage where the letter O stands for zero (`2O26`, `3,500.O0`) is corrected, with a warning
  - item names are normalized (`Widget A` → `WidgetA`, `WidgetA (rush order)` → `WidgetA` with note
    "rush order")
  - invoice numbers are normalized (`1002`, `INV 1012` → `INV-…`)
  - currency comes from markers on the amounts; mixed or conflicting currencies are rejected
- **Errors.** Unparseable *values* become `None` plus an `extraction_warning`. Unreadable *files* (missing,
  unsupported, truncated, non-UTF-8, entity bombs, excessive nesting, no invoice data) raise
  `IngestionError`.

## 5. Validation ([validation.py](../src/invoice_processor/validation.py))

| Check | Severity |
|---|---|
| Missing invoice number, vendor, total, or items | ERROR |
| Missing invoice or due date | WARNING |
| Values ingestion could not read or had to correct | WARNING (`extraction_uncertain`) |
| Quantity missing, zero, or negative | ERROR |
| Item unknown, zero stock, or combined quantity above stock | ERROR |
| Negative unit price or non-positive total | ERROR |
| Line total, subtotal, or total doesn't add up (±$0.01) | ERROR |
| Tax amount inconsistent with stated rate | WARNING (printed rates are often rounded) |
| Due date before invoice date | ERROR |
| Due date equals invoice date under Net terms | WARNING |
| Non-USD currency | WARNING |
| Pressure language (urgent, wire transfer, …) | WARNING (blocks approval at any amount) |

Inventory access goes through `InventoryLookup.get_stock_levels`: one parameterized query per invoice on a
read-only connection. A database failure raises, and the workflow marks the invoice failed. It never
reports an invoice as valid without having checked it.

## 6. Approval

`approve_invoice(invoice, validation, approver, policy)` runs any `Approver`, then calls
`enforce_hard_rules`.

Before that, `_well_formed` turns approver output that isn't a strictly valid `ApprovalResult` into REJECTED
(flag `malformed_approval`). `enforce_hard_rules` then overrides to REJECTED any approval that does one of
these:

- approves a hard failure (validation errors, or a missing or non-positive total)
- is for a different invoice number
- approved different terms

It also forces scrutiny on when policy requires it, and stamps the reviewed terms.

`ApprovalPolicy`:

| Setting | Value |
|---|---|
| Scrutiny threshold | strictly greater than $10,000 USD |
| USD exchange rates | USD only (unknown currencies always get scrutiny) |
| Warnings that block approval at any amount | `suspicious_content` |
| Under scrutiny | every other warning blocks, except `unsupported_currency` |

`LLMApprover` adds the draft → critique → revise loop described in the README. Its result can never be
more lenient than `RuleBasedApprover`'s.

## 7. Agents

**What counts as an agent here:** a component whose output is produced by the LLM, through its own prompt
and output structure, optionally with tools. Under that definition there are **three agents**, and the
approval agent also runs a **critic role**.

- **Agents run only when `build_workflow(..., llm=...)` is given a client.** In the CLI that means
  `--llm=on`, or `auto` with `XAI_API_KEY` set.
- **Agents never call each other.** LangGraph routes between nodes, and each agent runs inside one node.

| | Validation agent | Approval agent | Critic (role inside the approval agent) | Payment agent |
|---|---|---|---|---|
| **Implemented in** | `agents.review_validation` + `agent_tools.build_validation_tools` | `approval_agent.LLMApprover.review` (draft, `_revise`) | `LLMApprover._critique` (LLM call + `deterministic_findings`) | `agents.run_payment_agent` + `agent_tools.PaymentToolset` |
| **Invoked by** | `validate` node, only if an LLM is configured **and** deterministic validation passed | `approve` node → `approve_invoice` → `approver.review` (`build_workflow` selects `LLMApprover` when given an LLM and no approver) | `LLMApprover.review` loop | `pay` node → `payment_step`, only after `authorize_payment` succeeded |
| **Input** | Invoice JSON + `ValidationResult` JSON | Invoice, `ValidationResult`, policy text (not the policy engine's decision) | Same context + policy engine facts (baseline decision and reasoning, scrutiny explanation) + the draft | `PaymentAuthorization` only (invoice number, vendor, amount, currency) |
| **Output** | `ValidationReview` + `ToolCallRecord`s → `validation_review`, `tool_calls` | `ApprovalResult` incl. `review_trail` → `approval` | `Critique(findings)`, merged with deterministic findings | `PaymentResult` from the tool's execution record, `PaymentReport`, `ToolCallRecord`s |
| **Tools** | `lookup_inventory`, `check_stock` | none | none | `mock_payment` (bound) |
| **LLM requests** | `run_tools`: tool rounds + final structured answer | 1 draft + up to 2 revisions | 1 per round (≤ 2) | `run_tools`: ≤ 3 rounds + final structured answer |
| **On failure** | Deterministic result stands; `agent_errors` | Rule-based decision, flag `llm_unavailable` | (part of the approval agent) | Invoice not paid → `failed`; a payment already made is kept |

**Deliberately not agents** (deterministic):

- `extract_invoice` (ingestion)
- `validate_invoice` (validation checks)
- `RuleBasedApprover` (the policy engine; also produces the critic's facts and the leniency floor)
- `authorize_payment` / `execute_authorized_payment` / `DuplicatePaymentGuard`
- the `reject` node

**How the agents are connected:**

- **Nothing downstream reads the validation agent's review.** It is advisory and only appears in reports.
- **The approval agent works from the deterministic validation result.**
- **The payment agent sees only the authorization derived from the approved invoice.**
- **The reflection loop runs entirely inside the `approve` node.** It is recorded in `review_trail` and
  in the LLM call log.

### Execution flow (LLM mode)

```
ingest (deterministic) ──unreadable──► END failed
   │
   ▼
validate: validate_invoice (deterministic)
   │        └─ valid & LLM ─► Validation agent ── lookup_inventory / check_stock   (advisory)
   ├─ inventory unavailable ─► END failed
   ├─ invalid ─────────────────────────────────────────────► reject ─► END rejected
   ▼ valid
approve: RuleBasedApprover (baseline facts)
         Approval agent: draft ─► Critic ─► findings? ─yes─► revise ─┐   (≤ 2 rounds)
                                    ▲                                │
                                    └────────────────────────────────┘
         _finalize (never more lenient than baseline) ─► enforce_hard_rules
   ├─ rejected ────────────────────────────────────────────► reject ─► END rejected
   ▼ approved
pay: authorize_payment ──refused──► SKIPPED
   │
   ▼ PaymentAuthorization
   Payment agent ── mock_payment tool (bound) ─► DuplicatePaymentGuard ─► mock_payment() ─► END completed / failed
```

In deterministic mode the graph is the same, minus the three agents: `validate` runs only the
deterministic checks, `approve` uses `RuleBasedApprover`, and `pay` calls `execute_authorized_payment`
directly.

### Tool calls ([tools.py](../src/invoice_processor/tools.py), [agent_tools.py](../src/invoice_processor/agent_tools.py))

```
LLM ──tool call(name, json args)──► Toolset.invoke
                                      ├─ name in this agent's allowlist?
                                      ├─ args valid for the tool's strict Pydantic model?
                                      ├─ call budget remaining?
                                      ├─ run handler (refusals and exceptions become error results)
                                      └─ log + ToolCallRecord ──► ToolResult(ok, data | error) ──► LLM
```

- **Validation toolset.** `lookup_inventory` and `check_stock` on a read-only connection with
  parameterized SQL.
- **Payment toolset.** `mock_payment` bound to one `PaymentAuthorization`.
  - Requests that don't exactly match the authorization are refused.
  - A second payment is refused.
  - The budget is 3 calls.
  - It executes with the *authorized* values, never the model's copies.

The xAI tool loop ([llm/xai.py](../src/invoice_processor/llm/xai.py)) caps tool-calling rounds and then
asks for a final structured answer with `chat.parse`.

### LLM request telemetry ([llm/telemetry.py](../src/invoice_processor/llm/telemetry.py))

Every network request goes through `XAIClient._request` → `observe_request`. It logs one line immediately
before the request and one immediately after, on logger `invoice_processor.llm.calls`, and produces an
`LLMCallRecord` with:

- sequence number, agent, provider, model, and operation
- start time, completion time, and duration
- whether the API call succeeded, and whether the structured output parsed
- request ID, token usage, cost, and tool-call count
- a sanitized error, if any

No prompts, outputs, or keys are logged. Schema failures record only the failing field paths. The CLI
collects the records per invoice (`collect_llm_calls`) for the report's **LLM CALLS** table and the
JSON output. `--no-llm-log` hides successful requests, but failures are always shown.

## 8. Payment ([payment.py](../src/invoice_processor/payment.py))

```
authorize_payment(invoice, validation, approval) ──► PaymentAuthorization ──► execute_authorized_payment ──► pay()
          │ raises PaymentNotAuthorizedError with every blocker                            │
          └──────────────────────────────────────────────────────────────────────────────► SKIPPED
```

`payment_blockers` lists every reason not to pay:

- an invoice, validation result, or approval that isn't a genuine, strictly valid model instance (dicts,
  look-alike objects, and unvalidated `model_construct()` objects are refused)
- no approval, or a decision that isn't APPROVED
- the approval doesn't cover these terms
- validation failed
- the invoice, validation, and approval refer to different invoice numbers
- a missing invoice number or vendor, or a non-positive total

These are re-checked explicitly after the blockers, not with an `assert`, so the checks survive
`python -O`.

`DuplicatePaymentGuard` wraps the payment function and refuses a second PAID for the same invoice number.
Static tests in `test_qa_invariants.py` parse the source to enforce three things:

- only `execute_authorized_payment` (and the guard delegating to it) calls a payment function
- only `authorize_payment` constructs a `PaymentAuthorization`
- only three known places call `execute_authorized_payment`

## 9. Failure handling

| Failure | Behaviour | Outcome |
|---|---|---|
| File missing / unsupported / malformed / no invoice data | `IngestionError` | `failed`, batch continues |
| Field unreadable (`Due Date: yesterday`) | `None` + warning → validation warning | judged by policy |
| Ambiguous or conflicting currency | `IngestionError` | `failed` |
| Database missing, corrupt, or bad stock values | `InventoryDatabaseError` | `failed`, never validated |
| Validation agent fails (any exception) | Deterministic result stands, `agent_errors` recorded | unaffected |
| Approval agent fails or returns invalid output | Deterministic policy decision, flagged `llm_unavailable` | unaffected |
| Approver approves a hard failure, wrong invoice, or wrong terms | Overridden to REJECTED, flagged `approval_overridden` | `rejected` |
| Approver returns malformed output (dict, string, unvalidated object) | Treated as REJECTED, flagged `malformed_approval` | `rejected` |
| Caller passes forged `invoice` / `validation` / `approval` into `workflow.invoke()` | Dropped by `WorkflowInput` | unaffected |
| Any LLM request fails (API error or invalid structured output) | Logged at ERROR, shown under LLM FAILURES, end-of-run stderr warning | per agent row above |
| Payment agent fails before paying | FAILED payment, nothing paid | `failed` |
| Payment agent fails after paying | The tool's PAID record is kept | `completed` |
| Payment function raises | FAILED payment | `failed` |
| Same invoice number paid earlier in the run | FAILED, "duplicate payment blocked" | `failed` (exit 3) |
| Unexpected exception in a node | Caught per invoice by the CLI | `failed`, batch continues |

## 10. Module map

| Module | Responsibility |
|---|---|
| `models.py` | Contracts |
| `ingestion/` | `extract_invoice` and the format parsers |
| `database.py` | Schema, seed, `init_db`, `get_connection(read_only=)` |
| `inventory.py` | `InventoryLookup` and `SQLiteInventory` |
| `validation.py` | `validate_invoice` |
| `approval.py` | `ApprovalPolicy`, `RuleBasedApprover`, `approve_invoice`, `enforce_hard_rules` |
| `approval_agent.py` | `LLMApprover` and the critique loop |
| `payment.py` | `authorize_payment`, `mock_payment`, `process_payment`, `DuplicatePaymentGuard` |
| `tools.py` | `Tool`, `Toolset`, `ToolResult`, `ToolCallRecord` |
| `agent_tools.py` | Validation and payment toolsets |
| `agents.py` | `review_validation`, `run_payment_agent` |
| `llm/` | `LLMClient` interface, settings, `XAIClient`, provider registry, request telemetry (`telemetry.py`) |
| `workflow.py` | LangGraph graph and `payment_step` |
| `cli.py`, `report.py` | Command line, report rendering, JSON output, exit codes |
| `observability.py` | Invoice-tagged text and JSON logging |
