import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  Eye,
  FileText,
  Grid2X2,
  List,
  LoaderCircle,
  RefreshCw,
  Search,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";

import { useNavigate } from "react-router-dom";
import { Card, Button, StatusPill } from "../components/ui";
import { UploadDropzone } from "../components/UploadDropzone";
import { useDocumentStore } from "../store/documentStore";
import { formatBytes, formatDate } from "../utils/format";

const PRIMARY_FILTERS = ["ALL", "PDF", "DOCX", "XLSX", "CSV"];
const MORE_FILTERS = ["TXT"];

function documentType(document) {
  return (
    document.file_type ||
    document.filename?.split(".").pop() ||
    ""
  )
    .replace(".", "")
    .toUpperCase();
}

export default function Documents() {
  const navigate = useNavigate();
  const documents = useDocumentStore((state) => state.documents);
  const loading = useDocumentStore((state) => state.loading);
  const error = useDocumentStore((state) => state.error);
  const load = useDocumentStore((state) => state.load);
  const remove = useDocumentStore((state) => state.remove);
  const reindex = useDocumentStore((state) => state.reindex);

  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("ALL");
  const [viewMode, setViewMode] = useState("list");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [reindexingId, setReindexingId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [moreOpen, setMoreOpen] = useState(false);
  const moreRef = useRef(null);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const close = (event) => {
      if (moreRef.current && !moreRef.current.contains(event.target)) {
        setMoreOpen(false);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase();

    return documents.filter((document) => {
      const filename = document.original_filename || document.filename || "";
      const type = documentType(document);
      const matchesSearch =
        !term ||
        filename.toLowerCase().includes(term) ||
        type.toLowerCase().includes(term) ||
        String(document.status || "").toLowerCase().includes(term);
      const matchesType = filter === "ALL" || type === filter;
      return matchesSearch && matchesType;
    });
  }, [documents, search, filter]);

  const handleDelete = async (document) => {
    const name = document.original_filename || document.filename || "document";
    if (!window.confirm(`Delete ${name}? This removes the document and its vectors.`)) {
      return;
    }

    setDeletingId(document.id);
    try {
      await remove(document.id);
    } finally {
      setDeletingId(null);
    }
  };

  const handleReindex = async (document) => {
    setReindexingId(document.id);
    try {
      await reindex(document.id);
    } finally {
      setReindexingId(null);
    }
  };

  return (
    <div className="documents-page space-y-4 p-3 sm:p-5">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-end">
        <div className="flex items-center gap-2 self-end lg:self-auto">
          <div className="inline-flex rounded-xl border border-slate-800 bg-ink-900 p-1">
            <button
              type="button"
              aria-label="List view"
              onClick={() => setViewMode("list")}
              className={`grid h-9 w-10 place-items-center rounded-lg ${
                viewMode === "list"
                  ? "bg-violet-600 text-white"
                  : "text-slate-400 hover:bg-slate-800"
              }`}
            >
              <List className="h-4 w-4" />
            </button>
            <button
              type="button"
              aria-label="Grid view"
              onClick={() => setViewMode("grid")}
              className={`grid h-9 w-10 place-items-center rounded-lg ${
                viewMode === "grid"
                  ? "bg-violet-600 text-white"
                  : "text-slate-400 hover:bg-slate-800"
              }`}
            >
              <Grid2X2 className="h-4 w-4" />
            </button>
          </div>

          <Button onClick={() => setUploadOpen(true)}>
            <UploadCloud className="h-4 w-4" />
            Upload Document
          </Button>
        </div>
      </div>

      <Card className="p-3 sm:p-4">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-center">
          <div className="relative min-w-0 flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
            <input
              className="field h-11 pl-10"
              placeholder="Search files by name, type, or keyword..."
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {PRIMARY_FILTERS.map((item) => (
              <button
                type="button"
                key={item}
                onClick={() => setFilter(item)}
                className={`rounded-full border px-4 py-2 text-xs font-medium transition ${
                  filter === item
                    ? "border-violet-600 bg-violet-950/45 text-violet-200"
                    : "border-slate-800 bg-ink-900 text-slate-400 hover:border-slate-700 hover:text-white"
                }`}
              >
                {item === "ALL" ? "All" : item}
              </button>
            ))}

            <div className="relative" ref={moreRef}>
              <button
                type="button"
                onClick={() => setMoreOpen((value) => !value)}
                className="rounded-full border border-slate-800 bg-ink-900 px-4 py-2 text-xs font-medium text-slate-400 hover:border-slate-700"
              >
                More ▾
              </button>
              {moreOpen && (
                <div className="absolute right-0 z-30 mt-2 min-w-[120px] rounded-xl border border-slate-800 bg-ink-900 p-1.5 shadow-2xl">
                  {MORE_FILTERS.map((item) => (
                    <button
                      key={item}
                      type="button"
                      onClick={() => {
                        setFilter(item);
                        setMoreOpen(false);
                      }}
                      className="w-full rounded-lg px-3 py-2 text-left text-xs text-slate-300 hover:bg-slate-800"
                    >
                      {item}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>

      {error && (
        <div className="rounded-xl border border-rose-900/70 bg-rose-950/20 p-3 text-xs text-rose-400">
          {error}
        </div>
      )}

      <Card className="overflow-hidden">
        <div className="border-b border-slate-800 px-4 py-4 sm:px-5">
          <h2 className="text-sm font-bold text-white">Knowledge Repository</h2>
          <p className="mt-1 text-[10px] text-slate-500">
            Search and manage parsed document assets.
          </p>
        </div>

        {viewMode === "list" ? (
          <div className="table-scroll overflow-x-auto">
            <table className="w-full min-w-[920px] text-left">
              <thead>
                <tr className="border-b border-slate-800 text-[10px] uppercase tracking-wide text-slate-500">
                  {[
                    "Document Name",
                    "Type",
                    "Pages",
                    "Size",
                    "Uploaded On",
                    "Status",
                    "Actions",
                  ].map((heading) => (
                    <th key={heading} className="px-5 py-3 font-medium">
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {filtered.map((document) => (
                  <tr
                    key={document.id}
                    data-document-type={documentType(document)}
                    className="document-list-card border-b border-slate-800/75 text-[12px] text-slate-400 transition hover:bg-slate-900/35"
                  >
                    <td className="px-5 py-4">
                      <span className="document-card-copy flex min-w-0 items-center gap-3 font-semibold text-slate-200">
                        <FileText
                          data-document-type={documentType(document)}
                          className="document-card-icon document-list-icon h-4 w-4 shrink-0 text-violet-300"
                        />
                        <span className="document-card-title max-w-[360px] truncate">
                          {document.original_filename || document.filename}
                        </span>
                      </span>
                    </td>
                    <td className="px-5 py-4">{documentType(document)}</td>
                    <td className="px-5 py-4">{document.page_count || "—"}</td>
                    <td className="px-5 py-4">{formatBytes(document.file_size)}</td>
                    <td className="px-5 py-4">{formatDate(document.created_at)}</td>
                    <td className="px-5 py-4">
                      <StatusPill
                        status={
                          document.status ||
                          (document.ready_for_rag ? "ready" : "uploaded")
                        }
                      />
                    </td>
                    <td className="px-5 py-4">
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          className="h-9 w-9 p-0"
                          title="View document"
                          onClick={() => navigate(`/documents/${document.id}`)}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          className="h-9 w-9 p-0"
                          title="Re-index document"
                          disabled={reindexingId === document.id}
                          onClick={() => void handleReindex(document)}
                        >
                          {reindexingId === document.id ? (
                            <LoaderCircle className="h-4 w-4 animate-spin" />
                          ) : (
                            <RefreshCw className="h-4 w-4" />
                          )}
                        </Button>
                        <Button
                          variant="ghost"
                          className="h-9 w-9 p-0 text-rose-400"
                          title="Delete document"
                          disabled={deletingId === document.id}
                          onClick={() => void handleDelete(document)}
                        >
                          {deletingId === document.id ? (
                            <LoaderCircle className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="grid gap-3 p-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {filtered.map((document) => (
              <div
                key={document.id}
                data-document-type={documentType(document)}
                className="document-card-ui document-grid-card rounded-xl border border-slate-800 bg-ink-800 p-4"
              >
                <div className="flex items-start gap-3">
                  <div
                    data-document-type={documentType(document)}
                    className="document-card-icon-wrap document-type-icon grid h-10 w-10 shrink-0 place-items-center rounded-lg border border-violet-900/60 bg-violet-950/35 text-violet-300"
                  >
                    <FileText className="document-card-icon h-5 w-5" />
                  </div>
                  <div className="document-card-copy min-w-0 flex-1">
                    <div className="document-card-title truncate text-xs font-semibold text-slate-200">
                      {document.original_filename || document.filename}
                    </div>
                    <div className="document-card-meta mt-1 text-[9px] text-slate-500">
                      {documentType(document)} • {formatBytes(document.file_size)}
                    </div>
                  </div>
                </div>

                <div className="mt-4 flex items-center justify-between">
                  <StatusPill
                    status={
                      document.status ||
                      (document.ready_for_rag ? "ready" : "uploaded")
                    }
                  />
                  <div className="flex gap-1">
                    <Button variant="ghost" className="h-8 w-8 p-0" onClick={() => navigate(`/documents/${document.id}`)}>
                      <Eye className="h-3.5 w-3.5" />
                    </Button>
                    <Button variant="ghost" className="h-8 w-8 p-0" disabled={reindexingId === document.id} onClick={() => void handleReindex(document)}>
                      <RefreshCw className={`h-3.5 w-3.5 ${reindexingId === document.id ? "animate-spin" : ""}`} />
                    </Button>
                    <Button variant="ghost" className="h-8 w-8 p-0 text-rose-400" disabled={deletingId === document.id} onClick={() => void handleDelete(document)}>
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {!loading && !filtered.length && (
          <div className="py-16 text-center text-xs text-slate-600">
            No documents found.
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center gap-2 py-16 text-xs text-slate-500">
            <LoaderCircle className="h-4 w-4 animate-spin" />
            Loading repository…
          </div>
        )}
      </Card>

      {uploadOpen && (
        <div className="fixed inset-0 z-[80] grid place-items-center bg-slate-950/75 p-4 backdrop-blur-sm">
          <Card className="w-full max-w-[620px] p-4 sm:p-5">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-bold text-white">Upload Knowledge Assets</h3>
                <p className="mt-1 text-[10px] text-slate-500">
                  Upload up to 10 PDF, DOCX, TXT, CSV, or XLSX files. Maximum 25 MB per file.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setUploadOpen(false)}
                className="grid h-8 w-8 place-items-center rounded-lg border border-slate-800 text-slate-400 hover:text-white"
                aria-label="Close upload"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <UploadDropzone />
          </Card>
        </div>
      )}
    </div>
  );
}
