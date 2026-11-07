"""Read-only tests for TXT-only summary fixes. No DB, Ollama, API, or hardcoded production facts."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'backend/app/services/summary_service.py'
spec = importlib.util.spec_from_file_location('txt_summary_candidate', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Service = mod.SummaryService


def ev(n, content, *, filename='birch_transport.txt', section=''):
    return {'chunk_id': f'dummy-{n}', 'content': content, 'metadata': {
        'document_id': 'owned-demo', 'filename': filename, 'file_type': 'txt',
        'chunk_index': n, 'section': section,
    }}


def check(label, condition):
    assert condition, f'FAILED: {label}'
    print(f'PASS {label}')

# Invented numbers and names deliberately DIFFER from the live user's test corpus.
# A heading-only chunk is absent, mimicking indexers that discard short headings.
chunks = [
    ev(0, '''Birch Transport closed fiscal year 2024 with $91.2 million in revenue on
2,450 vehicles sold. Growth accelerated each quarter. Customer satisfaction was 4.4 out of 5.
The company reported manufacturing workforce of 210 employees.'''),
    ev(1, '''Quarter       Units Sold     Revenue (USD)
Q1 2024               500      $18,500,000
Q4 2024               720      $28,900,000
FY2024 Total        2,450      $91,200,000
Jan-Oct total units: 2,100
Jan-Oct total revenue: $78,000,000'''),
    ev(2, '''Product lineup comprised an SUV, sedan, and EV. Birch SUV Horizon was the top-selling model with 1,100 vehicles sold. The East region was the smallest market and West was the largest.'''),
    ev(3, '''Market analysis: industry competition intensified. Birch Transport average transaction price was $39,800 and customer satisfaction reached 4.4 out of 5. Inventory and production were managed carefully.'''),
    # FY2025 OUTLOOK heading deliberately missing; its body is present.
    ev(4, '''Management is targeting 9% unit sales growth for FY2025.
Planned priorities include:
- A 700-unit increase in EV production capacity.
- Opening three new regional service centers.
- Expansion in the East region.'''),
    ev(5, '''- Continued customer-experience investment aimed at moving CSAT from 4.4 toward a target of 4.9.
- Disciplined cost management across manufacturing and supply-chain operations.'''),
    ev(6, '''12. SOURCE FILES
1. birch_transport_financials.pdf
This report was assembled from verified source files.'''),
]
for mode in ('document', 'executive'):
    result = Service._grounded_narrative_report_summary(evidence=chunks, scope=mode)
    check(f'{mode} complete summary returned', result is not None)
    summary, citations = result
    overview = summary.split('Document Overview' if mode=='document' else 'Executive Overview', 1)[1].split('\n\n', 1)[0]
    check(f'{mode} year-end metrics lead', '91.2 million' in overview and '2,450' in overview)
    check(f'{mode} correct year versus Jan-Oct subtotals', '78,000,000' not in overview)
    for label, expected in [('EV capacity', '700-unit'), ('service centers','three new regional service centers'),
                            ('East expansion','Expansion in the East region'),
                            ('CSAT target','target of 4.9'), ('cost discipline','Disciplined cost management')]:
        check(f'{mode} {label}', expected in summary)
    check(f'{mode} provenance', any('Source: birch_transport.txt' in x for x in citations))

# The formatter must not invent year-end revenue when only a partial period is indexed.
partial = [ev(0, 'Jan-Oct total units: 2,100\nJan-Oct total revenue: $78,000,000\nIndustry transaction price and customer satisfaction increased.'),
           ev(1,'Product lineup and regional performance improved. Inventory in production rose in fiscal year 2024.'),
           ev(2,'Management is targeting 9% unit sales growth for FY2025 and plans more service centers and manufacturing jobs.')]
output = Service._grounded_narrative_report_summary(evidence=partial, scope='document')
check('no fabricated annual report totals', output is None or '$91,200,000' not in output[0])
check('custom section is untouched', Service._grounded_narrative_report_summary(evidence=chunks, scope='section') is None)

# Verify code-level scope: all existing methods retained, only one updated.
PREVIOUS = Path('/mnt/data/DocMindAI_7_Evaluation_Targeted_20261107/backend/app/services/summary_service.py')
if PREVIOUS.is_file():
    def members(path):
        cls = next(x for x in ast.parse(path.read_text()).body if isinstance(x, ast.ClassDef) and x.name == 'SummaryService')
        return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n,(ast.FunctionDef, ast.AsyncFunctionDef))}
    before, after = members(PREVIOUS), members(SOURCE)
    changed = [x for x in after if before.get(x) != after[x]]
    check('one method changed; no new or deleted methods', changed == ['_grounded_narrative_report_summary'] and set(before) == set(after))
    print('Source SHA256:', hashlib.sha256(PREVIOUS.read_bytes()).hexdigest())
print('Candidate SHA256:', hashlib.sha256(SOURCE.read_bytes()).hexdigest())
print('All TXT-only offline assertions passed. Live accuracy and latency NOT measured.')
