# DocMindAI — V2 24-test manual evaluation

**Input:** `24 testing DocMind AI V2.zip` and `docmindai_10_performance_results.json` (user-supplied).
**Status:** Measured quality of submitted results, NOT a verification of a new code candidate.
**Content rating:** 179/240 = **7.46/10** (12 Passed, 9 Partial, 3 Failed). These are manual rubric scores, not RAGAS scores.
**Timing:** 6/10 of the separately measured scenarios completed all three measured runs in <=10s; four structured summaries did not. Timing of the other 14 content tests not supplied.
**Citation note:** The summaries have file-level citations, not independently tested claim-level citation precision. Q15 returned no citation.

| # | Test | Module | Content /10 | Status | Accuracy/completeness/citations | Latency |
|---:|---|---|---:|---|---|---|
| 1 | Q6 — DOCX top-selling model | Chat & Ask | 8 | Partial | Correct models; missing Trailblazer 5,640 and Volt-X 1,850 sales figures. | N/A |
| 2 | Q7 — TXT FY2026 plans | Chat & Ask | 9 | Partial | 12%, 3,000 capacity, two centers, East, and CSAT 4.6→5.0 correct; cost-management point missing. | N/A |
| 3 | Q9 — TXT competitors | Chat & Ask | 8 | Partial | 38,100/36,900 and 4.6/4.2 correct; no explicit $1,200 and 0.4-point differences. | N/A |
| 4 | Q15 — XLSX workforce | Chat & Ask | 0 | Failed | Local model request failed; no facts/citations. Root cause not established without traceback. | N/A |
| 5 | Q10 — CSV Jan–Oct totals | Chat & Ask | 9 | Passed | 12,940 vehicles and $488,538,000 correctly reported; minor presentation glitch. | PASS; median 0.164s |
| 6 | PDF File Summary | Summaries | 3 | Failed | Inventory/competition described but Trailblazer 5,640 and $41,200 missing; broken headings/stray numbering. | N/A |
| 7 | PDF Executive Summary | Summaries | 2 | Failed | Missing major figures; miscategorized rival pricing under Outlook; truncated statements. | N/A |
| 8 | DOCX File Summary | Summaries | 5 | Partial | FY2025 and product units present; Q1, regional volumes, and outlook not complete; misleading CSAT superlative. | N/A |
| 9 | DOCX Executive Summary | Summaries | 5 | Partial | Annual totals, 12% and 3,000 capacity shown; no Q1/Q4 detail and regional/product figures; metadata in Challenges/Outlook. | N/A |
| 10 | TXT File Summary | Summaries | 4 | Partial | Annual totals correct, but no Jan–Oct figures, broken section headings and stray numbering. | N/A |
| 11 | TXT Executive Summary | Summaries | 7 | Partial | Annual totals, 12%, 3,000, CSAT target correct; East/two centers missing; citation file-level. | N/A |
| 12 | Monthly CSV File Summary | Summaries | 8 | Partial | 12,940 and $488,538,000, Jan/Oct correct; lacks computed Jan-Oct revenue difference and monthly trend. | N/A |
| 13 | Monthly CSV Executive Summary | Summaries | 8 | Partial | Totals and Jan/Oct correct; lacks explicit $18,488,000 growth and trend analysis. | N/A |
| 14 | Data XLSX File Summary | Summaries | 9 | Passed | Jan–Oct totals, average price, inventory and workforce correctly reported; verbose itemization. | N/A |
| 15 | Data XLSX Executive Summary | Summaries | 9 | Passed | Jan–Oct totals, stock/production, all workforce values correct; lacks concise business synthesis. | N/A |
| 16 | FY2025 CSV File Summary | Summaries | 9 | Passed | All required FY2025 metrics present with file citation. | FAIL; median 75.178s |
| 17 | FY2025 CSV Executive Summary | Summaries | 9 | Passed | All key FY2025 metrics correct; report remains metric-list oriented. | FAIL; median 39.265s |
| 18 | FY2026 XLSX File Summary | Summaries | 9 | Passed | All required FY2026 values present with file citation. | FAIL; median 77.114s |
| 19 | FY2026 XLSX Executive Summary | Summaries | 9 | Passed | All required FY2026 values present with file citation. | FAIL; median 30.158s |
| 20 | C2 — DOCX/TXT | Comparisons | 10 | Passed | Annual revenue/vehicles and Q1/Q4 revenue correct in both; two citations. | PASS; median 0.116s; warmup 12.853s (over 10s) |
| 21 | C3 — Monthly CSV/XLSX | Comparisons | 10 | Passed | Jan–Oct units and revenue equal; appropriate source sheets. | PASS; median 0.095s |
| 22 | C4 — TXT/XLSX | Comparisons | 10 | Passed | Volt-X stock 265, production 310, Manufacturing 520 equal. | PASS; median 0.115s |
| 23 | C6 — PDF/TXT | Comparisons | 10 | Passed | 38,100/41,500 prices and 4.6/4.4 CSAT correct. | PASS; median 0.103s |
| 24 | C7 — PDF/DOCX/TXT | Comparisons | 9 | Passed | 41,200 starting price and 5,640 sales match all three; extra source in text rather than table. | PASS; median 0.104s |

## Metrics status

- Faithfulness: qualitatively strong on observed comparison metrics, with inaccurate or unsupported wording in some summaries; no claim-level assessor trace.
- Context Recall: document facts absent in several generated summaries, but raw retrieval evidence not available to compute formal recall.
- Context Precision: some summary outputs mix metadata/irrelevant excerpts; chunk ranking quality not independently measured.
- Answer Relevance: strong in comparisons and structured metrics, weak on certain broad summaries.
- Graceful Refusal: not exercised in this V2 24-test ZIP; earlier 3 negative tests previously passed. No new formal score.

## Confirmed issues / correction boundaries

1. **Four slow structured summaries**: `summary_service.summarize` invokes SummaryAgent before its existing verified CSV/XLSX table presentation even when that presentation already contains the answer. Candidate moves this successful grounded path before the model call, with the same output keys; it does not change the underlying table parser.
2. **Q15 model request error**: the screenshot shows a failure but supplies no backend traceback. Root cause remains unverified; do not alter the Ollama or reasoning layer based on the screenshot alone.
3. **PDF/DOCX/TXT formatting and content omissions**: defects visibly involve formatting/executive finding selection. Current production `summary_agent.py` and a byte-verified November 1 frozen checkout are not provided in this turn; do not patch blindly.
4. **Existing passing comparisons**: preserve unchanged.

## Strict source provenance
The modified `summary_service.py` is derived from the accessible **November 5 targeted candidate**, SHA-256 22f7461cb3976386bd64d608960d492f7275883d7a64cdacc10683c965cdd5d7; the Nov 1 baseline is logical, not byte-verified. **Only install after local hash matches**, and retain backups. No API/schema/UI/auth/graph modifications.

## Validation required
- Static `compileall` and 4 offline guard checks passed; no actual Windows Ollama/FastAPI live run from this environment.
- Rerun four slow summaries at 3 measured requests each and check exact answer content/citations.
- Rerun all 24 V2 tests plus the original seven-regression baseline, and capture Q15 backend traceback if failure repeats.
- Check a novel uncached CSV/XLSX summary and a custom scoped summary to avoid an overfit fast path.
