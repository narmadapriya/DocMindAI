# DocMindAI — 7 Partial Tests, 2026-11-07

## Evidence
- Seven user-submitted exported PDF reports in `7 partial files testing results.zip`.
- User's PowerShell console results: 7 warmups and 21 measured API calls. All 28 reported HTTP success, answer present, citation present, and duration <=10 sec. There is no attached raw JSON latency report, server trace, or retrieved Chroma evidence.
- Benchmarked against the supplied source `apex_motors.txt`, `apex_motors_market_report.pdf`, and `apex_motors_annual_report.docx` where available.

## Content rating rubric
9–10: main requested facts complete, coherent and cited; 7–8: substantive core correct but notable omissions or noisy output; 5–6: partial coverage with significant omissions; <=4: misses an essential requested goal. Manual content assessment, **not RAGAS**.

| Original test | Score | Classification | Median measured seconds | Key evidence |
|---|---:|---|---:|---|
| Q9 TXT comparison | 8/10 | Partial | 0.378 | Correct $38,100 / $36,900 and 4.6/5 / 4.2/5; no $1,200 and 0.4-point comparison |
| PDF File Summary | 7/10 | Partial | 0.059 | Correct Trailblazer 5,640/$41,200 and Apex/Horizon figures; OCR chart line `Industry Industry Average =@=` and merged bullets |
| PDF Executive Summary | 7/10 | Partial | 0.052 | Correct reported figures and inventory outlook; OCR noise and no $3,400 / 0.2-point synthesis |
| DOCX File Summary | 9/10 | Pass | 0.046 | Correct FY2025 $612.4M/16,070, Q1 and Q4, South 5,150, East 3,180, Trailblazer 5,640, 1,240 workforce, FY2026 targets; minor duplication |
| DOCX Executive Summary | 9/10 | Pass | 0.045 | Correct fiscal and operational priorities; minor repetition |
| TXT File Summary | 4/10 | Fail | 0.051 | Starts with Jan–Oct $488,538,000/12,940 rather than full-year $612.4M/16,070; omits priority facts in sections 3–11 |
| TXT Executive Summary | 4/10 | Fail | 0.056 | Same fiscal-year omission and incomplete FY2026: 3,000-unit Volt-X, East region, CSAT 4.6→5.0 absent |

**Latest 7-test mean: 48/70 = 6.86/10.** Performance **7/7 cases passed** (<10 s across measured and warmup calls).

## Combined 13-case context
Six earlier-passing tests (not rerun in this package): Q6 10/10, Q7 9/10, Monthly CSV File 9/10, Monthly CSV Executive 9/10, XLSX File 9/10, XLSX Executive 9/10. Combined indicative total with seven new tests is (48+55)/130=**7.92/10**, but is not a fresh live 13-case result.

## Five dimensions
- Faithfulness: directly reported figures largely match source values; misleading emphasis on ten-month numbers is a completeness defect. No formal claim-level faithfulness score.
- Context Recall: summary output omits FY2025 and outlook facts available in the full TXT source, but retrieval recall cannot be assigned without selected evidence chunks.
- Context Precision: PDF chart OCR pollutes selected findings; numerical precision has not been measured against indexed chunks.
- Answer Relevance: Q9 / DOCX relevant; TXT summaries fail core all-period narrative.
- Graceful Refusal: not tested; no score.

## Root-cause scope and protection
1. **Confirmed** `backend/app/services/summary_service.py` method `_grounded_narrative_report_summary` accepts meaningless OCR chart-axis text as a fact (PDF summaries). Proposed correction excludes chart-axis artifacts and cleans embedded bullets/heading prefixes. Minimal, no RAG/backend/DB changes.
2. **Observed, exact runtime cause unresolved:** TXT summary missed full FY2025 facts and most FY2026 plan despite original TXT containing those facts. Running the latest packaged formatter offline on complete source chunks **does** return all required totals and 5 FY2026 priorities. Therefore source evidence selection, stale indexing, or a local-source mismatch is a possible cause, **not verified**. Need actual retrieved chunks + installed file hash. Do not change retrieval, model, database, or source documents without this evidence.
3. **Intentional frozen verifier restriction:** `backend/app/rag/agents/reasoning_agent.py`, `_txt_deterministic_reasoning`, adds derived price/CSAT differences only when question explicitly requests them, to comply with numeric-verification rules. Don't bypass or change without a verified safe reconciliation. Q9's original question is unchanged.

## Source version
The PDF-only patch was prepared from previous `DocMindAI_10Test_Targeted_Candidate` summary_service.py (SHA256 ecd5b89e...), not a byte-verified November 1 frozen baseline. Installation must be refused if local SHA differs. Do not freeze this candidate until real content and latency regressions pass.
