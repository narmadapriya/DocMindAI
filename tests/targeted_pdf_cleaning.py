"""Read-only method-level guard for source-derived narrative formatting."""
from pathlib import Path
import importlib.util
import re

path=Path(__file__).resolve().parents[1]/'backend/app/services/summary_service.py'
spec=importlib.util.spec_from_file_location('candidate_summary_service',path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
S=mod.SummaryService

def item(text, n):
    return {'chunk_id':f'p{n}','content':text,'metadata':{'filename':'synthetic_market.pdf','file_type':'pdf','page_number':n}}

items=[
 item('South Coast Electric reported quarterly revenue gains and higher units sold. The market expanded at a mid-single-digit pace; satisfaction remained high.',1),
 item('South Coast Electric offers a product lineup spanning SUVs and sedans. Fleet SUV Eclipse\nSUV\nStarting at $43,500\nFY2026 units sold: 1,520',2),
 item('Competitive Landscape\nManufacturer\nAvg. Price (USD)\nCSAT (out of 5)\nEV Share of Sales\nSouth Coast Electric\n$39,800\n4.5\n12.0%\nWest Ridge Motors\n$42,100\n4.4\n9.0%',3),
 item('Key Market Drivers • More electric demand. • SUV sales remained strong. Industry Industry Average =@= South Coast Electric 2022 2023 2024 2025 2026.',4),
 item('Inventory Snapshot\nThe electric crossover reached its reorder threshold. Manufacturing has scheduled more production for FY2027, and management plans growth in western regions.',5),
]
for scope in ('document','executive'):
    result=S._grounded_narrative_report_summary(evidence=items,scope=scope)
    assert result is not None,scope
    text,citations=result
    assert '$43,500' in text and '1,520' in text,(scope,text)
    assert 'Industry Industry' not in text and '=@=' not in text,(scope,text)
    assert 'Source: synthetic_market.pdf' in '\n'.join(citations),scope
    assert '•' not in text,(scope,text)
    print('PASS',scope,'chart artifacts removed and product numbers retained')
print('2/2 targeted tests passed')
