import React from "react";
import { Check, FileSpreadsheet, FileText } from "lucide-react";
import { useDocumentStore } from "../store/documentStore";
import { cn } from "../utils/cn";
import { documentReady } from "../utils/format";
export function DocumentSelector({ multiple = true, max, }) {
    const documents = useDocumentStore((s) => s.documents);
    const selected = useDocumentStore((s) => s.selectedIds);
    const setSelected = useDocumentStore((s) => s.setSelected);
    const select = (id) => {
        if (!multiple)
            return setSelected([id]);
        if (selected.includes(id))
            return setSelected(selected.filter((value) => value !== id));
        if (max && selected.length >= max)
            return;
        setSelected([...selected, id]);
    };
    const readyDocs = documents.filter(documentReady);
    return (<div className="space-y-2">
      {readyDocs.map((doc) => {
            const active = selected.includes(doc.id);
            const ExcelIcon = (doc.file_type || doc.filename).toLowerCase().includes("xls")
                ? FileSpreadsheet
                : FileText;
            const docType = String(
                doc.file_type ||
                (doc.original_filename || doc.filename || "document").split(".").pop() ||
                "document"
            ).replace(".", "").toUpperCase();
            return (<button key={doc.id} type="button" data-document-type={docType} aria-pressed={active} onClick={() => select(doc.id)} className={cn("document-card-ui flex w-full items-center gap-2 rounded-lg border px-3 py-2.5 text-left transition", active
                    ? "border-violet-600 bg-violet-950/45"
                    : "border-slate-800 bg-ink-800 hover:border-slate-700")}>
            <ExcelIcon className={cn("document-card-icon h-4 w-4", active ? "text-violet-300" : "text-cyan-400")}/>
            <span className="document-card-copy min-w-0 flex-1">
              <span className="document-card-title block truncate text-xs font-semibold text-slate-200">
                {doc.original_filename || doc.filename}
              </span>
              <span className="document-card-meta document-card-light-only mt-0.5 hidden text-[9px] text-slate-500">
                {doc.chunk_count ? `${doc.chunk_count} chunks indexed` : "RAG-ready document"}
              </span>
            </span>
            {active ? (<span className="document-card-select grid h-4 w-4 place-items-center rounded-full bg-violet-600">
                <Check className="h-3 w-3 text-white"/>
              </span>) : (<span className="document-card-select h-4 w-4 rounded-full border border-slate-600"/>)}
          </button>);
        })}
      {!readyDocs.length && (<p className="rounded-lg border border-dashed border-slate-800 p-3 text-center text-xs text-slate-500">
          No RAG-ready documents yet.
        </p>)}
    </div>);
}
