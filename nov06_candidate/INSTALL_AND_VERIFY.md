# Candidate installation — NOT an approved frozen-baseline patch

Only modified app file: `backend/app/services/summary_service.py`.

**Source provenance:** 2026-11-05 candidate SHA-256 `22f7461cb3976386bd64d608960d492f7275883d7a64cdacc10683c965cdd5d7`.
**Proposed replacement SHA-256:** `a09cae3733117717c895e4600d79951f9a7ec49c362740c2dae13f2cd9e58e05`.
The true final Nov-1 frozen checkout was never byte-verified. Do NOT overwrite a local file with a different hash.

## Windows PowerShell from project root
```powershell
$Target = ".\backend\app\services\summary_service.py"
(Get-FileHash $Target -Algorithm SHA256).Hash.ToLower()
```
The printed hash must equal the **source provenance** above. If it doesn't, stop and supply your current file for a matching minimal patch.

Back up then copy only the corrected source file from this extracted package:
```powershell
Copy-Item $Target "$Target.pre_nov06_backup"
Copy-Item ".\_candidate\backend\app\services\summary_service.py" $Target
python -m compileall -f .\backend\app\services\summary_service.py
```
Replace `_candidate` with the directory where you extracted the ZIP. Restart FastAPI, then test all affected and previously passed modes. Roll back by copying `.pre_nov06_backup` over the file if needed.

## Offline guard test
```powershell
python .\_candidate\tests\test_summary_latency_guard.py
```
## Live performance tests
Re-run your existing `docmindai_10_performance_only.py` from the project root with `--runs 3 --warmup 1`, then repeat all 24 content cases and the original 7-case baseline regression. Q15 still needs a fresh backend traceback if it fails.
