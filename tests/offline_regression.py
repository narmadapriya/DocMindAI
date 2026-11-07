"""Offline source-fact regression; no network, Ollama, app imports, or DB access."""
from pathlib import Path
import ast
import importlib.util
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'backend/app/services/summary_service.py'
compile(PATH.read_text(encoding='utf8'), str(PATH), 'exec')
spec=importlib.util.spec_from_file_location('local_docmind_summary', PATH)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
S=mod.SummaryService
count=0

def check(name, condition):
    global count
    assert condition, name
    print('PASS', name)
    count+=1

def ev(filename, content, index=0, **extra):
    meta={'filename':filename, 'file_type':filename.rsplit('.',1)[-1],
          'document_id':'isolated-test','chunk_index':index, **extra}
    return {'chunk_id':f'c{index}', 'content':content, 'metadata':meta}

# Deliberately invented, non-Apex fixture: tests general source extraction,
# not hardcoding the user's benchmark values.
report='sample_annual.docx'
doc=[
    ev(report,'Arbor Motors closed fiscal year 2024 with $91.2 million in revenue on 2,450 vehicles sold.',0),
    ev(report, 'Quarter | Units Sold | Revenue (USD)\nQ1 2024 | 500 | $18,500,000\nQ4 2024 | 720 | $28,900,000\nFY2024 Total | 2,450 | $91,200,000',1,chunk_type='table'),
    ev(report,'Model | Body Type | Starting Price | FY2024 Units\nArbor SUV Explorer | SUV | $45,200 | 1,100\nArbor EV Glide | EV | $50,000 | 600',2,chunk_type='table'),
    ev(report,'Region | Units Sold | Share of FY2024 Sales\nNorth | 500 | 20.4%\nSouth | 900 | 36.7%\nWest | 700 | 28.6%\nEast | 350 | 14.3%',3,chunk_type='table'),
    ev(report,'Manufacturing workforce has 210 employees. Management is targeting 8% unit sales growth for FY2025. Planned regional service expansion will accompany EV production investment.',4),
]
for scope in ('document','executive'):
    out=S._grounded_narrative_report_summary(evidence=doc,scope=scope)
    check(f'{scope} DOCX parses source tables',out is not None)
    if out:
        txt,cites=out
        for label,number in [('Q1','500'),('Q4','720'),('SUV','1,100'),('South','900'),('East','350')]:
            check(f'{scope} DOCX {label} {number}',number in txt)
        check(f'{scope} DOCX no flattened pipe dump','Quarter | Units' not in txt and 'Model | Body' not in txt)

pdf='sample_market.pdf'
pdf_chunks=[ev(pdf,'Arbor Motors introduced its new product lineup while revenue grew ahead of the industry. The market comparison showed SUV growth and stronger customer satisfaction.',0,page_number=1),
            ev(pdf,'Product Catalog\nArbor SUV Explorer\nSUV\nStarting at $45,200\nFY2024 units sold: 1,100\nArbor EV Glide\nElectric Hatchback\nStarting at $50,000\nFY2024 units sold: 600',1,page_number=2),
            ev(pdf,'Competitive Landscape\nManufacturer\nAvg. Price (USD)\nCSAT (out of 5)\nEV Share of Sales\nArbor Motors\n$41,200\n4.5\n12.0%\nRidge Motor Co.\n$46,800\n4.1\n10.0%',2,page_number=3),
            ev(pdf,'Inventory Snapshot\nThe EV model is close to its reorder threshold as of October 2024. OCR text: j8%# unverified chart pixels. The company plans additional production in FY2025. Conclusion\nArbor Motors enters FY2025 with higher growth than the wider market.',3,page_number=4),
            ev(pdf,'Industry average revenue remained flat. The North region leads expansion and production capacity is expected to increase next year.',4,page_number=5)]
for scope in ('document','executive'):
    out=S._grounded_narrative_report_summary(evidence=pdf_chunks,scope=scope)
    check(f'{scope} PDF returns complete narrative',out is not None)
    if out:
        txt,cites=out
        check(f'{scope} PDF no OCR noise','j8%#' not in txt and 'OCR text:' not in txt)
        check(f'{scope} PDF provenance',any('Page' in c for c in cites))
        check(f'{scope} PDF source product details', '$45,200' in txt and '1,100' in txt)

# TXT fixture: both yearly and partial-year values must remain distinct.
txtfile='sample_consolidated.txt'
txtchunks=[
    ev(txtfile,'FY2024 Total: 2,450 vehicles sold; revenue $91,200,000.\nJan-Oct total units: 2,100\nJan-Oct total revenue: $78,000,000',0),
    ev(txtfile,'Quarter       Units Sold       Revenue (USD)\nQ1 2024          500        $18,500,000\nQ4 2024          720        $28,900,000\nFY2024 Total     2,450        $91,200,000',1),
    ev(txtfile,'Product Catalog\nArbor SUV Explorer\nSUV\nStarting at $45,200\nFY2024 units sold: 1,100\nArbor EV Glide\nElectric Hatchback\nStarting at $50,000\nFY2024 units sold: 600',2),
    ev(txtfile,'Market Analysis: Arbor Motors beat industry averages. Average transaction price $41,200. Customer satisfaction 4.5 out of 5.',3),
    ev(txtfile,'FY2025 OUTLOOK\nManagement is targeting 8% unit sales growth for FY2025.\nPlanned priorities include:\n- A 500-unit increase in EV production capacity.\n- Opening three new regional service centers.\n- Expansion in the East region.\n- Improving customer satisfaction from 4.5 to 4.8.\n- Disciplined cost management across supply-chain operations.',4,section='FY2025 OUTLOOK'),
]
for scope in ('document','executive'):
    out=S._grounded_narrative_report_summary(evidence=txtchunks,scope=scope)
    check(f'{scope} TXT accepts fully grounded broad summary',out is not None)
    if out:
        text,cites=out
        check(f'{scope} TXT full-year lead','2,450' in text.split('Financial')[0].split('Key Metrics')[0])
        check(f'{scope} TXT no separators','====' not in text)
        check(f'{scope} TXT document provenance',any('sample_consolidated.txt' in c for c in cites))

# Negative: no summary of unrelated or incomplete evidence should trigger.
check('Unrelated simple PDF falls back',S._grounded_narrative_report_summary(evidence=[ev('tech.pdf','The rain is cold.',1)],scope='document') is None)
check('Custom section summary falls back',S._grounded_narrative_report_summary(evidence=doc,scope='section') is None)
check('No mixed-file summary direct fast path',S._grounded_narrative_report_summary(evidence=pdf_chunks+doc,scope='document') is None)

# Review change scope compared to already-installed November 6 candidate.
baseline=Path('/mnt/data/DocMindAI_13Test_Targeted_Fix_20261106/backend/app/services/summary_service.py')
if baseline.is_file():
    orig=ast.parse(baseline.read_text()); updated=ast.parse(PATH.read_text())
    def funcs(tree):
        cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SummaryService')
        return {x.name:ast.dump(x,include_attributes=False) for x in cl.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
    f0,f1=funcs(orig),funcs(updated)
    changed=[k for k in f1 if f0.get(k)!=f1[k]]
    check('Only grounded narrative method changed',changed==['_grounded_narrative_report_summary'])

print(f'{count}/{count} offline checks passed. End-to-end latency NOT measured.')
