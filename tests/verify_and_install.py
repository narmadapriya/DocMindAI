"""Checksum-gated replacement of exactly one summary-only file.

Default mode is READ ONLY. Never edit any other project file.
"""
from pathlib import Path
import argparse
import hashlib
import shutil
from datetime import datetime
import sys

REL='backend/app/services/summary_service.py'
SOURCE_SHA='ecd5b89ecff7ec8952a5a6cadde47460dd8f6306ed786d43d430ac52b162687e'
CANDIDATE_SHA='d9b865d3686e41ca7d0dd44983b4921e3cb58c3b0e59c7c21f82d2dcc28fc233'

def hashfile(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description='Safe DocMindAI summary-only patch guard')
    parser.add_argument('--project-root',required=True,help='Existing DocMindAI root')
    parser.add_argument('--install',action='store_true',help='Create backup and replace only when SHA matches')
    args=parser.parse_args()
    root=Path(args.project_root).expanduser().resolve()
    target=root/REL
    candidate=Path(__file__).resolve().parents[1]/REL
    if not target.is_file() or not candidate.is_file():
        sys.exit('Missing current source or candidate. No changes made.')
    candidate_hash=hashfile(candidate)
    if candidate_hash!=CANDIDATE_SHA:
        sys.exit('Candidate file checksum mismatch. No changes made.')
    current=hashfile(target)
    print('TARGET:',target)
    print('CURRENT_SHA256:',current)
    print('CANDIDATE_SHA256:',candidate_hash)
    if current==CANDIDATE_SHA:
        print('ALREADY_INSTALLED: No replacement required')
        return
    if current!=SOURCE_SHA:
        sys.exit('UNKNOWN_SOURCE_VERSION: Do not overwrite. No changes made.')
    print('SAFE_TO_INSTALL: installed source matches the audited November 6 candidate')
    if not args.install:
        print('READ_ONLY verification complete; no files changed')
        return
    backup=target.with_name(target.name+'.pre7_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.bak')
    shutil.copy2(target,backup)
    try:
        shutil.copy2(candidate,target)
        if hashfile(target)!=CANDIDATE_SHA:
            raise RuntimeError('Copied file checksum mismatch')
        compile(target.read_text(encoding='utf-8'),str(target),'exec')
    except Exception:
        shutil.copy2(backup,target)
        raise
    print('INSTALLED:',target)
    print('BACKUP:',backup)
    print('Syntax verified. Runtime behavior requires separate testing.')
if __name__=='__main__': main()
