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
```

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

That function overrides to REJECTED any approval that does one of these:

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

## 7. Tools and agents ([tools.py](../src/invoice_processor/tools.py), [agent_tools.py](../src/invoice_processor/agent_tools.py), [agents.py](../src/invoice_processor/agents.py))

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

## 8. Payment ([payment.py](../src/invoice_processor/payment.py))

```
authorize_payment(invoice, validation, approval) ──► PaymentAuthorization ──► execute_authorized_payment ──► pay()
          │ raises PaymentNotAuthorizedError with every blocker                            │
          └──────────────────────────────────────────────────────────────────────────────► SKIPPED
```

`payment_blockers` lists every reason not to pay:

- no approval, or a decision that isn't APPROVED
- the approval doesn't cover these terms
- validation failed
- the invoice, validation, and approval refer to different invoice numbers
- a missing invoice number or vendor, or a non-positive total

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
| `llm/` | `LLMClient` interface, settings, `XAIClient`, provider registry |
| `workflow.py` | LangGraph graph and `payment_step` |
| `cli.py`, `report.py` | Command line, report rendering, JSON output, exit codes |
| `observability.py` | Invoice-tagged text and JSON logging |
