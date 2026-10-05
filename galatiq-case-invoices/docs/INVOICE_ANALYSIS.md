# Sample invoice analysis

This analysis of the 20 files in `data/invoices/` was done **before** writing extraction or validation
code. It drove the parser design and the test expectations: `tests/test_ingestion.py` and
`tests/test_validation.py` assert exactly these results. The stock check uses the seed inventory: WidgetA
15, WidgetB 10, GadgetX 5, FakeItem 0.

## Edge-case matrix

| Invoice | Format | Extracted (vendor · total · due · items) | Validation | Reason / extraction hazards |
|---|---|---|---|---|
| **1001** | TXT | Widgets Inc. · $5,000.00 · 2026-02-01 · WidgetA×10, WidgetB×5 | ✅ pass | Clean baseline. |
| **1002** | TXT | Gadgets Co. · $15,000.00 · 2026-01-30 · GadgetX×20 | ❌ stock | 20 > 5 in stock. Misspelled labels (`INVOCE`, `Vndr`, `Dt`, `Itms`, `Amt`, `Pymnt`). ID is `1002` with no prefix. Due date equals the invoice date despite Net 30 (warning). |
| **1003** | TXT | Fraudster LLC · $100,000.00 · *unparseable* · FakeItem×100 | ❌ zero stock | FakeItem has 0 stock. Due date is the word `yesterday`. Urgent wording asking for a wire transfer. |
| **1004** | JSON | Precision Parts Ltd. · $1,890.00 · 2026-02-22 · WidgetA×3, WidgetB×2 | ✅ pass | Clean. |
| **1004 revised** | JSON | same vendor · $5,940.00 · + GadgetX×5 | ✅ pass; ⚠ duplicate ID | `revision: R1`, same invoice number. GadgetX 5 of 5 is exactly at the limit. Never paid twice. |
| **1005** | JSON | Global Supply Chain Partners · $15,225.00 · 2026-03-18 · 3 items | ❌ stock | GadgetX 8 > 5. WidgetB 10 of 10 is exactly at the limit. |
| **1006** | CSV (key/value) | Acme Industrial Supplies · $2,750.00 · 2026-02-10 · WidgetA×5, WidgetB×3 | ✅ pass | Repeated `item` keys: a naive dict parse would silently lose an item. |
| **1007** | CSV (one row per item) | MegaWidgets Corp · stated $15,525.00 · 2026-02-28 · 3 items | ❌ stock + math | WidgetA 20 > 15 and WidgetB 15 > 10. **Total is wrong by $110** ($14,750 + $885 = $15,635). US dates. |
| **1008** | TXT (email) | NoProd Industries · $9,900.00 · 2026-01-20 · SuperGizmo×12, MegaSprocket×6 | ❌ unknown items | Email headers around the data; `From:` must not be read as the vendor. |
| **1009** | JSON | *(empty)* · −$250.00 · *null* · WidgetA×−5, WidgetB×2 | ❌ integrity | Negative quantity and total, empty vendor, null due date. Subtotal ($1,000) contradicts the lines (−$250). |
| **1010** | TXT (table) | Consolidated Materials Group · $7,185.00 · 2026-02-26 · 4 lines | ✅ pass | `WidgetA (rush order)` at a different price; combined WidgetA is 12 of 15. Includes $150 shipping. Dates written out (`January 27, 2026`). |
| **1011** | TXT + PDF | Summit Manufacturing Co. · $3,000.00 · 2026-02-20 · WidgetA×6, WidgetB×3 | ✅ pass | The PDF has no subtotal or tax lines. Same invoice in two files. |
| **1012** | TXT + PDF | QuickShip Distributers · $9,975.00 · 2026-02-25 · 3 items | ✅ pass (2 warnings) | Scan damage: `2O26`, `$3,500.O0`, `Widget A`, `Gadget X`, `INV 1012`. Total just under $10K. |
| **1013** | JSON + PDF | Atlas Industrial Supply · stated $22,562.80 · 2026-03-24 · 8 lines | ❌ stock + math | Each line alone is within stock; **combined**, WidgetA 22 > 15, WidgetB 18 > 10, GadgetX 9 > 5. **Total is $50 too high**. |
| **1014** | XML | TechParts International · €4,125.00 · 2026-02-26 · WidgetA×4, WidgetB×6 | ✅ pass (currency warning) | Only EUR invoice; no exchange rate, so it gets additional scrutiny. |
| **1015** | CSV (one row per item) | Reliable Components Inc. · $6,500.00 · 2026-02-28 · 3 items | ✅ pass | Same layout as 1007 but with ISO dates. |
| **1016** | JSON | Widgets Inc. · $3,233.00 · 2026-02-27 · 3 items | ❌ unknown item | WidgetC is not in inventory. |

## Where the brief and the files disagree

1. **Formats.** The brief lists PDF, CSV, JSON, and TXT; `invoice_1014.xml` also exists, and is supported.
2. **INV-1004 has a revised version** with the same invoice number and a higher total. The brief lists only
   the original as a normal pass.
3. **Files not in the brief's scenario table:** 1005, 1007, 1010–1015, and the 1004 revision. Of these,
   1005, 1007, and 1013 fail.
4. **Wrong stated totals.** 1007 is $110 low and 1013 is $50 high. The 1013 error is deliberate:
   `data/generate_pdfs.py` adds `+ 50` to the grand total.
5. **INV-1009 has more problems than the brief lists:** besides the negative quantity, an empty vendor, a
   null due date, and contradictory totals.
6. **The brief's run example** names `data/invoices/invoice1.txt`, which does not exist.

## Decisions taken from this analysis

- Repeated lines for the same item are added together before checking stock; quantity equal to stock
  passes.
- Stock is fixed per run, not decremented across invoices, so results don't depend on processing order.
- An invoice number may be paid only once per run; revisions and copies need manual reconciliation.
- Stated totals are kept as written and checked against the line arithmetic (±$0.01).
- Non-USD invoices get scrutiny until an exchange rate is configured.
