import { FileText, Search, X } from "lucide-react";
import React, { useMemo, useState } from "react";
import { Card, Button } from "./ui";
import { normalizeCitation } from "../utils/format";

function sourceIconTone(filename = "") {
  const name = String(filename).toLowerCase();

  if (name.endsWith(".pdf")) return "text-red-500";

  if (
    name.endsWith(".xlsx") ||
    name.endsWith(".xls") ||
    name.endsWith(".csv")
  ) {
    return "text-green-600";
  }

  if (name.endsWith(".docx") || name.endsWith(".doc")) {
    return "text-blue-600";
  }

  return "text-slate-500";
}

function sourceDocumentType(filename = "") {
  const name = String(filename || "document");
  return String(name.split(".").pop() || "document").replace(".", "").toUpperCase();
}

function sourceLocation(citation) {
  const parts = [];

  if (citation.page_number != null) {
    parts.push(`Page ${citation.page_number}`);
  }

  if (citation.section_name) {
    parts.push(`Section ${citation.section_name}`);
  }

  if (citation.sheet_name) {
    parts.push(`Sheet ${citation.sheet_name}`);
  }

  if (citation.table_name) {
    parts.push(`Table ${citation.table_name}`);
  }

  return parts.join(" • ") || "Indexed source";
}

export function CitationList({ citations, onOpen, heading = "Sources" }) {
  if (!citations.length) return null;

  return (
    <div className="mt-4 space-y-2 border-t border-slate-800 pt-3">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
        {heading}
      </div>

      {citations.map((raw, index) => {
        const citation = normalizeCitation(raw, index);

        return (
          <button
            key={citation.id || index}
            onClick={() => onOpen?.(citation)}
            data-document-type={sourceDocumentType(
              citation.filename || citation.document_name || citation.label,
            )}
            className="flex w-full items-start gap-2 rounded-lg border border-slate-800 bg-ink-800 p-2.5 text-left hover:border-violet-800"
          >
            <FileText
              className={`document-card-icon mt-0.5 h-3.5 w-3.5 ${sourceIconTone(
                citation.filename || citation.document_name || citation.label,
              )}`}
            />

            <span className="min-w-0 flex-1">
              <span className="block truncate text-[11px] font-medium text-slate-300">
                {citation.filename ||
                  citation.label ||
                  `Source ${index + 1}`}
              </span>

              <span className="block text-[9px] text-slate-500">
                {sourceLocation(citation)}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

export function RagSourcesPanel({
  documents,
  selectedIds,
  onToggle,
}) {
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase();

    if (!term) return documents;

    return documents.filter((doc) =>
      (doc.original_filename || doc.filename || "")
        .toLowerCase()
        .includes(term),
    );
  }, [documents, search]);

  return (
    <aside className="chat-source-drawer hidden w-[320px] shrink-0 border-l border-slate-800 bg-[#03091a] lg:block">
      <div className="flex h-[52px] items-center justify-between border-b border-slate-800 px-3">
        <span className="text-xs font-semibold text-slate-200">
          RAG Sources Panel
        </span>

        <span className="rounded-full border border-emerald-900 bg-emerald-950/30 px-2 py-0.5 text-[9px] font-semibold text-emerald-400">
          Index Ready
        </span>
      </div>

      <div className="p-3">
        <div className="relative">
          <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-600" />

          <input
            className="field pl-9 text-xs"
            placeholder="Search indexed files..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>

        <div className="mt-4 text-[9px] font-semibold uppercase tracking-wider text-slate-500">
          Active Knowledge Base ({documents.length})
        </div>

        <div className="mt-2 space-y-2">
          {filtered.map((doc) => {
            const selected = selectedIds.some(
              (id) => String(id) === String(doc.id),
            );

            return (
              <button
                key={doc.id}
                onClick={() => onToggle(doc.id)}
                data-document-type={String(doc.file_type || (doc.original_filename || doc.filename || "document").split(".").pop() || "document").replace(".", "").toUpperCase()}
                aria-pressed={selected}
                className={`document-card-ui flex w-full items-center gap-2 rounded-lg border p-2.5 text-left ${
                  selected
                    ? "border-violet-700 bg-violet-950/30"
                    : "border-slate-800 bg-ink-800 hover:border-slate-700"
                }`}
              >
                <FileText
                  className={`document-card-icon h-4 w-4 shrink-0 ${sourceIconTone(
                    doc.original_filename || doc.filename,
                  )}`}
                />

                <div className="document-card-copy min-w-0 flex-1">
                  <div className="document-card-title truncate text-[11px] font-semibold text-slate-200">
                    {doc.original_filename || doc.filename}
                  </div>

                  <div className="document-card-meta text-[9px] text-slate-500">
                    {doc.chunk_count
                      ? `${doc.chunk_count} chunks indexed`
                      : "Repository document"}
                  </div>
                </div>

                <span
                  className={`document-card-select h-4 w-4 rounded-full border ${
                    selected
                      ? "border-emerald-400 bg-emerald-500"
                      : "border-slate-600"
                  }`}
                />
              </button>
            );
          })}

          {!filtered.length && (
            <div className="rounded-lg border border-slate-800 bg-ink-800 p-3 text-[10px] leading-5 text-slate-500">
              {documents.length
                ? "No indexed files match this search."
                : "No documents are available. Upload a document from Documents."}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}

export function CitationViewer({
  citation,
  loading = false,
  onClose,
}) {
  if (!citation) return null;

  const score =
    citation.confidence_score != null
      ? Number(citation.confidence_score)
      : citation.relevance_score != null
        ? Number(citation.relevance_score)
        : null;

  const metadata = citation.metadata || {};
  const pipeline = citation.rag_pipeline_details || {};
  const text = citation.chunk_text || citation.content || null;

  return (
    <div className="citation-viewer-backdrop fixed inset-0 z-50 grid place-items-center bg-[#020617]/90 p-4 backdrop-blur-sm">
      <Card className="citation-metasheet w-full max-w-[760px] overflow-hidden">
        <div className="citation-metasheet-header flex items-center justify-between border-b border-slate-800 px-4 py-3 sm:px-5 sm:py-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Search className="h-4 w-4 text-violet-300" />
            Vector Chunk Metasheet
          </div>

          <Button variant="ghost" className="citation-viewer-close" onClick={onClose}>
            <X className="h-4 w-4" />
            Close Viewer
          </Button>
        </div>

        <div className="citation-metasheet-body space-y-4 p-4 sm:p-5">
          <div className="flex items-start gap-3">
            <div
              data-document-type={sourceDocumentType(
                citation.document_name || citation.filename,
              )}
              className="document-card-icon-wrap citation-document-icon-tile grid h-10 w-10 shrink-0 place-items-center rounded-lg border border-emerald-800 bg-emerald-950/40"
            >
              <FileText
                className={`document-card-icon h-5 w-5 ${sourceIconTone(
                  citation.document_name || citation.filename,
                )}`}
              />
            </div>

            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-semibold text-slate-200">
                {citation.document_name ||
                  citation.filename ||
                  "N/A"}
              </div>

              <div className="mt-1 flex flex-wrap gap-x-2 gap-y-1 text-[10px] text-slate-500">
                <span>ID: {citation.chunk_id || "N/A"}</span>
                <span>
                  • Page:{" "}
                  {citation.page_number ??
                    citation.page ??
                    "N/A"}
                </span>

                {citation.sheet_name && (
                  <span>• Sheet: {citation.sheet_name}</span>
                )}

                <span>
                  • Confidence Index:{" "}
                  {Number.isFinite(score)
                    ? score.toFixed(2)
                    : "N/A"}
                </span>
              </div>
            </div>
          </div>

          {loading ? (
            <div className="rounded-xl border border-slate-800 bg-ink-800 p-5 text-center text-xs text-slate-500">
              Fetching chunk text and metadata from the backend…
            </div>
          ) : (
            <>
              <div>
                <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  Verbatim Indexed Text
                </div>

                <div className="rounded-xl border border-slate-800 bg-ink-800 p-4 text-sm italic leading-6 text-slate-300">
                  {text || "N/A"}
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-ink-800 p-4">
                <div className="text-xs font-semibold text-slate-300">
                  Chunk Metadata
                </div>

                <div className="mt-3 grid gap-2 text-[10px] text-slate-500 sm:grid-cols-2">
                  <div>
                    Chunk Type:{" "}
                    <span className="text-slate-300">
                      {citation.chunk_type || "N/A"}
                    </span>
                  </div>

                  <div>
                    Vector ID:{" "}
                    <span className="break-all text-slate-300">
                      {citation.vector_id ||
                        metadata.vector_id ||
                        "N/A"}
                    </span>
                  </div>

                  <div>
                    Source:{" "}
                    <span className="text-slate-300">
                      {citation.source || "N/A"}
                    </span>
                  </div>

                  <div>
                    Chunk Index:{" "}
                    <span className="text-slate-300">
                      {citation.chunk_index ??
                        metadata.chunk_index ??
                        "N/A"}
                    </span>
                  </div>

                  {Object.entries(metadata)
                    .filter(
                      ([key]) =>
                        ![
                          "vector_id",
                          "chunk_index",
                          "chunk_type",
                        ].includes(key),
                    )
                    .map(([key, value]) => (
                      <div key={key} className="break-words">
                        {key}:{" "}
                        <span className="text-slate-300">
                          {value == null
                            ? "N/A"
                            : typeof value === "object"
                              ? JSON.stringify(value)
                              : String(value)}
                        </span>
                      </div>
                    ))}
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-ink-800 p-4">
                <div className="text-xs font-semibold text-slate-300">
                  RAG Pipeline Information
                </div>

                <div className="mt-3 grid gap-2 text-[10px] text-slate-500 sm:grid-cols-2">
                  <div>
                    PostgreSQL Chunk:{" "}
                    <span className="text-slate-300">
                      {pipeline.postgres_chunk == null
                        ? "N/A"
                        : pipeline.postgres_chunk
                          ? "Yes"
                          : "No"}
                    </span>
                  </div>

                  <div>
                    Citation Persisted:{" "}
                    <span className="text-slate-300">
                      {pipeline.citation_persisted == null
                        ? "N/A"
                        : pipeline.citation_persisted
                          ? "Yes"
                          : "No"}
                    </span>
                  </div>

                  <div>
                    Vector Metadata Found:{" "}
                    <span className="text-slate-300">
                      {pipeline.vector_metadata_found == null
                        ? "N/A"
                        : pipeline.vector_metadata_found
                          ? "Yes"
                          : "No"}
                    </span>
                  </div>

                  <div>
                    Vector Store:{" "}
                    <span className="text-slate-300">
                      {pipeline.vector_store || "N/A"}
                    </span>
                  </div>
                </div>
              </div>

              {citation._detail_error && (
                <div className="rounded-xl border border-rose-900/70 bg-rose-950/20 p-3 text-[10px] text-rose-300">
                  {citation._detail_error}
                </div>
              )}
            </>
          )}
        </div>
      </Card>
    </div>
  );
}
