"""Read-only isolated regression for the two targeted complete source files.
No backend server, network access, live credentials, DB writes or Ollama calls.
"""
import ast
import importlib.util
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
service_path = ROOT / 'backend/app/services/summary_service.py'
reason_path = ROOT / 'backend/app/rag/agents/reasoning_agent.py'

for path in (service_path, reason_path):
    compile(path.read_text(encoding='utf8'), str(path), 'exec')

spec = importlib.util.spec_from_file_location('standalone_summary_service', service_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
SummaryService = mod.SummaryService

# Stub imports only for isolated ReasoningAgent class loading. These are not
# application changes and never connect to Ollama.
for name in ['app','app.rag','app.rag.llm','app.core']:
    sys.modules.setdefault(name, types.ModuleType(name))
llm = types.ModuleType('app.rag.llm.ollama_client')
llm.OllamaClient = type('OllamaClient', (), {})
llm.OllamaModelError = type('OllamaModelError', (Exception,), {})
sys.modules['app.rag.llm.ollama_client'] = llm
log = types.ModuleType('app.core.logging')
log.get_logger = lambda x: None
log.log_event = lambda *a, **kw: None
sys.modules['app.core.logging'] = log
perf = types.ModuleType('app.core.performance')
perf.TTLCache = type('TTLCache', (), {'__class_getitem__': classmethod(lambda cls, _:cls)})
sys.modules['app.core.performance'] = perf
spec = importlib.util.spec_from_file_location('standalone_reasoning_agent', reason_path)
rmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rmod)
ReasoningAgent = rmod.ReasoningAgent

count = 0

def check(label, cond):
    global count
    assert cond, label
    count += 1
    print('PASS', label)

# Q6: selected DOCX only, models from narrative and units from a separate
# source table. The code uses source values and IDs, not hardcoded facts.
q6 = 'Which vehicle model sold the most units in FY2025, and which showed the fastest year-over-year growth?'
meta = {'file_type':'docx','filename':'example_annual.docx'}
narr = {'chunk_id':'narrative', 'metadata':meta,
        'content': "Apex SUV Trailblazer was the company's top seller. Apex EV Volt-X posted the fastest year-over-year growth."}
table = {'chunk_id':'table', 'metadata':meta,
         'content': ('Model | Body Type | Starting Price | FY2025 Units\n'
                     'Apex SUV Trailblazer | SUV | $42,000 | 6,100\n'
                     'Apex EV Volt-X | EV | $46,000 | 1,900')}
r = ReasoningAgent._document_fact_fast_reasoning(q6, [narr, table])
check('Q6 recognizes both model identities', bool(r and 'Trailblazer' in r['answer'] and 'Volt-X' in r['answer']))
check('Q6 includes exactly supplied model counts', bool(r and '6,100' in r['answer'] and '1,900' in r['answer']))
check('Q6 cites narrative and table chunk', bool(r and set(r['used_evidence_ids']) == {'narrative','table'}))
r_missing = ReasoningAgent._document_fact_fast_reasoning(q6, [narr])
check('Q6 does not invent missing model figures', bool(r_missing and '6,100' not in r_missing['answer']))

# Use one complete business doc with independent topical sections. Entries
# deliberately use different business names and amounts to detect hardcoding.
meta_pdf = {'filename':'river_market.pdf','file_type':'pdf','page_number':2}
source = [
 {'chunk_id':'s1','metadata':meta_pdf, 'content':'Fiscal year 2024 total revenue was $91,000,000 from 2,400 units sold. Market share improved in the North region.'},
 {'chunk_id':'s2','metadata':meta_pdf, 'content':'The product lineup of SUV and EV models expanded. Falcon SUV was the top-selling model with 1,200 units sold. Starting price $42,500.'},
 {'chunk_id':'s3','metadata':meta_pdf, 'content':'Competitor average transaction price was $45,600. Customer satisfaction CSAT was 4.2 out of 5. The EV share grew.'},
 {'chunk_id':'s4','metadata':meta_pdf, 'content':'Inventory at the manufacturing facility was adequate. The company is planning an EV production expansion for fiscal year 2025.'},
]
for scope in ('document','executive'):
    res = SummaryService._grounded_narrative_report_summary(evidence=source, scope=scope)
    check(f'{scope}: complete unrelated business report accepted', res is not None)
    if res:
        answer,cites = res
        check(f'{scope}: source figures included', '$91,000,000' in answer and ('1,200' in answer or '2,400' in answer))
        check(f'{scope}: page citation retained', any('Page 2' in x for x in cites))
        check(f'{scope}: no raw markdown headings', '###' not in answer and '**' not in answer)

nonbusiness = [{'chunk_id':str(i),'metadata':{'filename':'guide.pdf','file_type':'pdf'},
                'content':'Chapter on woodland birds and geology.'} for i in range(5)]
check('Unrelated reports retain original agent', SummaryService._grounded_narrative_report_summary(evidence=nonbusiness, scope='document') is None)
check('Custom scope remains on original path', SummaryService._grounded_narrative_report_summary(evidence=source, scope='section') is None)
check('No missing evidence fabricated', SummaryService._grounded_narrative_report_summary(evidence=[], scope='executive') is None)

class ForbiddenAgent:
    def summarize(self, **kw):
        raise AssertionError('Complete grounded summary incorrectly called local model')

svc = SummaryService(summary_agent=ForbiddenAgent(), retrieval_graph=None)
svc._whole_document_evidence = lambda **kw: source
for scope in ('document','executive'):
    r=svc.summarize(user_id='u', document_ids=['d'], scope=scope)
    check(f'{scope}: result schema preserved', set(['summary','citations','scope','evidence_count','document_count']).issubset(r))
    check(f'{scope}: model bypass only for complete evidence', bool(r['summary'] and r['citations']))

print(f'OFFLINE CHECKS PASSED: {count}')
