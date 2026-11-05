"""Read-only, standard-library test of summary latency short-circuit. Run from project root."""
from pathlib import Path
import importlib.util

p=Path(__file__).resolve().parents[1] / "backend/app/services/summary_service.py"
spec=importlib.util.spec_from_file_location("summary_guard",p)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class StopAgent:
    def summarize(self, **kwargs):
        raise AssertionError("SummaryAgent should not run when verified tabular evidence is sufficient")

class SimpleAgent:
    def summarize(self, **kwargs):
        return {"summary":"ordinary path", "citations":[], "scope":kwargs.get("scope"),
                "evidence_count":len(kwargs["evidence"]), "document_count":1}
    def _focus_summary_evidence(self, evidence, instructions):
        return evidence

content="\n".join(
    "Row %d: Month: %s; Year: 2025; Units Sold: %d; Revenue (USD): %d" %
    (i,month,100+i,10000+i*100)
    for i,month in enumerate(("Jan","Feb","Mar","Apr"),1)
)
rows=[{"chunk_id":"chunk-1", "metadata":{"filename":"sample.csv", "file_type":"csv",
       "document_id":"document-1", "user_id":"user-1"}, "content":content}]
svc=module.SummaryService(StopAgent(), None)
svc._whole_document_evidence=lambda **kwargs: rows
for scope,instructions in [
    ("document","Generate a File Summary of the selected document."),
    ("executive","Generate an Executive Summary with key highlights and critical KPIs.")]:
    result=svc.summarize(user_id="user-1", document_ids=["document-1"],
                         scope=scope, scope_instructions=instructions)
    assert set(result)=={"summary","citations","scope","evidence_count","document_count"}
    assert "January" in result["summary"] and result["document_count"]==1
    print("PASS:",scope,"single-doc indexed CSV bypass")

non_tab=module.SummaryService(SimpleAgent(),None)
non_tab._whole_document_evidence=lambda **kwargs: [{
    "chunk_id":"chunk-2","metadata":{"filename":"sample.docx","file_type":"docx","document_id":"document-1"},
    "content":"A complete non-tabular document reports a fact supported by sufficient evidence."}]
assert non_tab.summarize(user_id="user-1",document_ids=["document-1"],scope="document")["summary"]
print("PASS: non-tabular unchanged model path")

scoped=module.SummaryService(SimpleAgent(),None)
scoped._whole_document_evidence=lambda **kwargs: rows
result=scoped.summarize(user_id="user-1",document_ids=["document-1"],scope="document",
                        scope_instructions="Focus solely on staff morale.")
assert "January" not in result["summary"]
print("PASS: custom scoped request uses existing agent path")
