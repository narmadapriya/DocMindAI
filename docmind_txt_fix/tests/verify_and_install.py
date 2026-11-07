"""Read-only by default; guard one-file DocMindAI TXT summary replacement."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, shutil, sys
from datetime import datetime

REL = 'backend/app/services/summary_service.py'
# Latest November 7 candidate confirmed in the preceding evaluation, NOT
# a byte-verified snapshot of the approved November 1 logical baseline.
KNOWN_LOCAL_SHA256 = 'd9b865d3686e41ca7d0dd44983b4921e3cb58c3b0e59c7c21f82d2dcc28fc233'
CORRECTED_SHA256 = '03124f7a86cf0000f0aebe31326b5eb0d4f62a15918505a72ac2cdbabc8ac97f'

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--install', action='store_true', help='Backup and replace only with exact known hashes')
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    candidate = Path(__file__).resolve().parents[1] / REL
    target = root / REL
    if not candidate.is_file() or not target.is_file():
        sys.exit('MISSING_SOURCE: no files changed')
    candidate_hash = sha(candidate)
    installed_hash = sha(target)
    print('PROJECT_FILE:', target)
    print('CURRENT_SHA256:', installed_hash)
    print('CANDIDATE_SHA256:', candidate_hash)
    if candidate_hash != CORRECTED_SHA256:
        sys.exit('BAD_CANDIDATE_SHA256: no files changed')
    if installed_hash == CORRECTED_SHA256:
        print('ALREADY_INSTALLED: no changes required')
        return
    if installed_hash != KNOWN_LOCAL_SHA256:
        sys.exit('UNKNOWN_SOURCE_VERSION: preserve current file; no files changed')
    print('SAFE_TO_INSTALL: exact approved November 7 candidate found')
    if not args.install:
        print('READ_ONLY: source remains unchanged')
        return
    backup = target.with_name(target.name + '.pre_txt_fix_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.bak')
    shutil.copy2(target, backup)
    try:
        shutil.copy2(candidate, target)
        if sha(target) != CORRECTED_SHA256:
            raise RuntimeError('hash mismatch after copy')
        compile(target.read_text(encoding='utf-8'), str(target), 'exec')
    except Exception:
        shutil.copy2(backup, target)
        raise
    print('INSTALLED:', target)
    print('BACKUP:', backup)
    print('Compiled: YES; live retest required')

if __name__ == '__main__':
    main()
