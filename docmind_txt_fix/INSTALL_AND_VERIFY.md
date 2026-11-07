# DocMindAI — TXT File & Executive Summary Only

**What this candidate changes:** only `backend/app/services/summary_service.py`, and within it only `_grounded_narrative_report_summary`. In `.txt` source-grounded whole-file narrative summaries, explicitly labelled fiscal-year figures receive priority over monthly/ten-month subtotals. The method also carries an explicitly dated outlook/priorities context across adjacent source chunks, including when a heading-only chunk has been discarded. Source text/numbers are extracted dynamically; there are no Apex-specific answers in the production code.

**Scope protections:** no changes to Chat & Ask, Comparisons, SummaryAgent, retrieval ranking, APIs, LangGraph contracts, UI, security, database, or CSV/XLSX handling. PDF/DOCX paths execute the same preexisting method conditions. The correct 10-second API speed target has already been achieved in the last submitted run, so this candidate makes no performance configuration changes.

**Source provenance caveat:** This complete Python file is derived from the **latest November 7 installed candidate** (known SHA256 `d9b865d3686e41ca7d0dd44983b4921e3cb58c3b0e59c7c21f82d2dcc28fc233`), **not a byte-verified copy** of the November 1 logical frozen baseline. If your installed source differs, DO NOT force this package over it. Provide the actual current file to reconcile the patch. The November 1 freeze is left unchanged.

## 1. Extract ZIP in Windows PowerShell

Place `DocMindAI_TXT_Two_Failures_Fix_20261107.zip` in `C:\Users\Admin\Desktop\PROJECTS\DocMindAI`:

```powershell
cd C:\Users\Admin\Desktop\PROJECTS\DocMindAI
Expand-Archive -LiteralPath ".\DocMindAI_TXT_Two_Failures_Fix_20261107.zip" -DestinationPath ".\docmind_txt_fix" -Force
```

## 2. Required read-only hash check

```powershell
python .\docmind_txt_fix\tests\verify_and_install.py --project-root .
```

Proceed only if `SAFE_TO_INSTALL`, or stop if already installed. If `UNKNOWN_SOURCE_VERSION`, do **not** overwrite the current source.

## 3. Install with automatic timestamped backup, compile, and run offline regression

```powershell
python .\docmind_txt_fix\tests\verify_and_install.py --project-root . --install
python -m compileall -f .\backend\app\services\summary_service.py
python .\docmind_txt_fix\tests\txt_two_failures_offline.py
python .\docmind_txt_fix\tests\offline_existing_regressions.py
python .\docmind_txt_fix\tests\existing_pdf_regression.py
```

## 4. Restart FastAPI

```powershell
cd C:\Users\Admin\Desktop\PROJECTS\DocMindAI\backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## 5. Retest **ONLY the two failed TXT cases** first

In DocMindAI Summaries, select only **`apex_motors.txt`**.

**File Summary**, use your original File Summary mode/prompt. **Executive Summary**, use your original Executive Summary mode/prompt. Both answers must:
- Lead with full FY2025 **$612.4 million revenue** and **16,070 vehicles**, sourced to the TXT *FY2025 Executive Summary* or *Quarterly Financial Highlights*.
- Separately identify January–October sales of **12,940 vehicles** and **$488,538,000**, with no conflation of reporting periods.
- Include FY2026 **12% unit-growth**, **3,000-unit Volt-X capacity**, **two new service centers**, **East-region expansion**, **CSAT from 4.6 toward 5.0**, and **disciplined costs** only when each is present in the indexed TXT evidence.
- Show TXT source citations, no 502/errors, no unsupported values, and request-to-response latency <=10 seconds.

Use the existing performance script as a **read-only 7-case regression** (7 warmups + 21 measured):

```powershell
cd C:\Users\Admin\Desktop\PROJECTS\DocMindAI
python .\tests\docmindai_7_live_performance.py `
  --base-url http://127.0.0.1:8000 `
  --runs 3 `
  --warmup 1 `
  --target 10 `
  --timeout 30 `
  --save-answers `
  --output .\docmindai_7_after_txt_fix.json
```

Verify saved full text/citations manually. A successful HTTP response does not establish factual accuracy. Also rerun the previously passing DOCX File/Executive, PDF File/Executive, Q6/Q7/Q9, CSV/XLSX, Q15 and Q10/C2/C3/C4/C6/C7 cases before approving any new freeze.

## If TXT still fails

Do not repeat source edits blindly: the selected Chroma/Postgres chunks may lack the FY total or outlook bullet bodies, or the installed version may differ. Capture your current `summary_service.py` SHA256, JSON answers (with no passwords/tokens), `evidence_count`, and a redacted read-only listing of the indexed TXT chunks. The formatter cannot ethically invent missing indexed evidence.
