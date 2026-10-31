r"""Non-invasive XLSX scoped-summary evidence diagnostic for DocMindAI.

Run from the project root (Windows PowerShell):
  $env:PYTHONPATH = (Resolve-Path .\backend).Path
  python .\tests\docmindai_summary_evidence_diagnostic.py \
      --xlsx .\fixtures\apex_motors_data.xlsx

Reads the real XLSX using existing parser, and loads the EXISTING installed
SummaryAgent source without importing the entire LangGraph package. Does not
change files, database, embeddings, or any RAG model. Does not call Ollama.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import sys

FOCUS = (
    'Summarize FY2025 sales performance, January–October revenue and units, '
    'vehicle inventory, and workforce distribution. Include exact numerical '
    'figures and key findings.'
)
TARGETS = {
    'Jan-Oct vehicle units': '12940',
    'Jan-Oct revenue': '488538000',
    'Volt-X name': 'volt-x',
    'Volt-X inventory 265': '265',
    'Volt-X production 310': '310',
    'Manufacturing workforce 520': '520',
}

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xlsx',type=Path,required=True)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=ap.parse_args()
    root=args.root.resolve();backend=root/'backend'
    sys.path.insert(0,str(backend))
    from app.rag.parser.excel_parser import parse_xlsx
    agent_file=backend/'app/rag/agents/summary_agent.py'
    spec=importlib.util.spec_from_file_location('docmindai_summary_diagnostic', agent_file)
    if spec is None or spec.loader is None: raise RuntimeError('Cannot load '+str(agent_file))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    parsed=parse_xlsx(args.xlsx)
    evidence=[{'chunk_id':str(i), 'content':item.text,
               'metadata': {'chunk_index':i, 'sheet_name':str(item.section or '')}}
              for i,item in enumerate(parsed.text_blocks)]
    focused=module.SummaryAgent._focus_summary_evidence(evidence,FOCUS)
    all_text='\n'.join(str(x['content']) for x in focused).lower()
    summary,_=module.SummaryAgent()._executive_summary(focused)
    print('Input XLSX: ', args.xlsx)
    print('Extracted text blocks:',len(evidence),'Focused:',len(focused))
    print('\nRequired facts present in selected evidence:')
    for name,value in TARGETS.items():
        print(f'  {"YES" if value in all_text else "NO ":3s}  {name}')
    print('\nExisting deterministic fallback output (no Ollama called):')
    print(summary)
    print('\nCoverage of reference fact tokens in fallback output:')
    for name,value in TARGETS.items():
        print(f'  {"YES" if value in summary.lower() else "NO ":3s}  {name}')
    print('\nThis is a SOURCE-EVIDENCE diagnostic, not a live inference or latency test.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
