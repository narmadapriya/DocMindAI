# DocMindAI — 7-Test PDF-only Candidate

This is a conditional corrective candidate. It is **not** an approved frozen baseline and does not claim verified 10/10 accuracy. The only modified production file is `backend/app/services/summary_service.py` and the only changed method is `_grounded_narrative_report_summary`.

## 1. Extract ZIP into project root

```powershell
cd C:\Users\Admin\Desktop\PROJECTS\DocMindAI
Expand-Archive -LiteralPath ".\DocMindAI_7_Evaluation_Targeted_20261107.zip" -DestinationPath ".\docmind_7_candidate" -Force
```

## 2. Read-only hash compatibility check

```powershell
python .\docmind_7_candidate\tests\verify_and_install.py --project-root .
```

Continue **only if** `SAFE_TO_INSTALL`. If `UNKNOWN_SOURCE_VERSION`, do not overwrite: the user's on-disk file is not the audited source.

## 3. Controlled replacement (optional; target PDF OCR only)

```powershell
python .\docmind_7_candidate\tests\verify_and_install.py --project-root . --install
python -m compileall -f .\backend\app\services\summary_service.py
python .\docmind_7_candidate\tests\offline_regression.py
python .\docmind_7_candidate\tests\targeted_pdf_cleaning.py
```

The installer creates a timestamped backup and restores it on verification failure. No app source is changed by the read-only check or offline tests.

## 4. Restart backend

```powershell
cd C:\Users\Admin\Desktop\PROJECTS\DocMindAI\backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## 5. Repeat tests

From root, run the already-installed `tests\docmindai_7_live_performance.py` with `--runs 3 --warmup 1 --target 10 --save-answers`, capture outputs and compare **every answer's content and source citations**. Retest previous passing Q6/Q7/monthly CSV/XLSX summaries, Q15 and Q10/C2/C3/C4/C6/C7 to prevent regressions. Scope excludes Comparisons modifications.

## 6. TXT diagnosis blocker

Compare the actual local `backend/app/services/summary_service.py` SHA256 to `MANIFEST.json`. Then inspect returned summary's `evidence_count` and server-side retrieved chunks for `apex_motors.txt`, especially FY2025 Executive Summary, Quarterly Financial Highlights, and FY2026 Outlook. Use read-only inspection; do not reindex or delete data automatically.

The 7-test console proves HTTP latency, **not** a 10/10 correctness rating. Performance speed is already safely below 10s in this test; changing timeout/model settings is not justified.
