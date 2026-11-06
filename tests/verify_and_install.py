r"""Safely verify/install only two known-version DocMindAI corrections.

Read-only verification:
    python .\docmind_13_candidate\tests\verify_and_install.py --project-root .
Install (after matching):
    python .\docmind_13_candidate\tests\verify_and_install.py --project-root . --install

Source matching uses semantic Python AST hashes so harmless CRLF/comment changes
will not force a false mismatch. Unknown behavior remains protected.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((CANDIDATE_ROOT/'MANIFEST.json').read_text(encoding='utf-8'))


def ast_sha256(path: Path) -> str:
    source = path.read_text(encoding='utf-8-sig')
    return hashlib.sha256(ast.dump(ast.parse(source), include_attributes=False).encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path('.'))
    parser.add_argument('--install',action='store_true',help='Back up and install only if all source versions match')
    args = parser.parse_args()
    project = args.project_root.resolve()
    plans = []
    for rel, record in MANIFEST['files'].items():
        installed = project / rel
        candidate = CANDIDATE_ROOT / rel
        if not installed.is_file() or not candidate.is_file():
            raise RuntimeError(f'Missing file: {installed} or {candidate}')
        source_ast = ast_sha256(installed)
        candidate_ast = ast_sha256(candidate)
        if candidate_ast != record['candidate_semantic_ast_sha256']:
            raise RuntimeError(f'Candidate integrity mismatch: {candidate}')
        if source_ast == candidate_ast:
            state='already_installed'
        elif source_ast == record['source_semantic_ast_sha256']:
            state='safe_to_install'
        else:
            raise RuntimeError(f'Unknown local version: {rel}; no files replaced. Upload your exact local file for a matching correction.')
        print(f'{rel}: {state}')
        plans.append((installed,candidate,state))
    if not args.install:
        print('READ-ONLY verification complete. Add --install to back up and replace compatible source files.')
        return 0
    for installed,candidate,state in plans:
        if state=='already_installed':
            continue
        tag=datetime.now().strftime('%Y%m%d_%H%M%S')
        backup=installed.with_name(installed.name+f'.pre_13test_{tag}.bak')
        shutil.copy2(installed,backup)
        shutil.copy2(candidate,installed)
        compile(installed.read_text(encoding='utf-8'),str(installed),'exec')
        print('INSTALLED:',installed,'BACKUP:',backup)
    print('Installation complete. Restart the backend, then run live 13-test and previous regressions.')
    return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('STOP:',exc,file=sys.stderr)
        sys.exit(1)
