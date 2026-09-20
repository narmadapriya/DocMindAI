import React, {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  Eye,
  FileText,
  Search,
} from "lucide-react";

import {
  Button,
  Card,
} from "../components/ui";

import {
  CitationViewer,
} from "../components/CitationPanel";

import {
  citationService,
} from "../services/citations";

import {
  getErrorMessage,
} from "../services/api";



function citationDocumentType(citation) {
  const filename = citation.document_name || citation.filename || "document";
  return String(filename.split(".").pop() || "document").replace(".", "").toUpperCase();
}

export default function Citations() {
  const [
    citations,
    setCitations,
  ] = useState([]);

  const [
    query,
    setQuery,
  ] = useState("");

  const [
    selected,
    setSelected,
  ] = useState(null);

  const [
    viewerLoading,
    setViewerLoading,
  ] = useState(false);

  const [
    loading,
    setLoading,
  ] = useState(true);

  const [
    error,
    setError,
  ] = useState("");


  useEffect(() => {
    let active = true;

    void citationService
      .list(150)
      .then(
        (items) => {
          if (!active) {
            return;
          }

          setCitations(
            items
          );

          setError("");
        },
      )
      .catch(
        (requestError) => {
          if (!active) {
            return;
          }

          setError(
            getErrorMessage(
              requestError,
            ),
          );
        },
      )
      .finally(
        () => {
          if (active) {
            setLoading(
              false
            );
          }
        },
      );

    return () => {
      active = false;
    };
  }, []);


  // ========================================================
  // Frontend Citation Deduplication Safety Check
  // ========================================================
  //
  // Backend already performs the same check.
  //
  // This is intentionally repeated here immediately before
  // filtering/rendering so duplicate cards cannot appear even
  // if an older/cached response reaches the browser.
  //
  // Exact key:
  //
  // document_id + page_number + chunk_id
  //
  // Citation content, scores, metadata and ordering are not
  // modified.
  // ========================================================

  const uniqueCitations =
    useMemo(
      () => {
        const seen =
          new Set();

        return citations.filter(
          (citation) => {
            const documentId =
              citation.document_id;

            const pageNumber =
              (
                citation.page_number
                ??
                citation.page
                ??
                null
              );

            const chunkId =
              citation.chunk_id;

            // Keep incomplete legacy/non-persisted citations.
            // Do not accidentally collapse unrelated items.
            if (
              !documentId
              ||
              !chunkId
            ) {
              return true;
            }

            const key =
              `${String(
                documentId,
              )}::${
                pageNumber == null
                  ? ""
                  : String(
                      pageNumber,
                    )
              }::${String(
                chunkId,
              )}`;

            if (
              seen.has(
                key
              )
            ) {
              return false;
            }

            seen.add(
              key
            );

            return true;
          },
        );
      },
      [
        citations,
      ],
    );


  // ========================================================
  // Search
  // ========================================================

  const filtered =
    useMemo(
      () => {
        const term =
          query
            .trim()
            .toLowerCase();

        if (!term) {
          return (
            uniqueCitations
          );
        }

        return (
          uniqueCitations.filter(
            (citation) =>
              [
                citation.filename,
                citation.document_name,
                citation.chunk_id,
                citation.quoted_text,
                citation.page_number,
              ]
                .filter(
                  (value) =>
                    value != null,
                )
                .join(" ")
                .toLowerCase()
                .includes(
                  term
                ),
          )
        );
      },
      [
        uniqueCitations,
        query,
      ],
    );


  // ========================================================
  // Context Viewer
  // ========================================================

  const openContext =
    async (
      citation,
    ) => {
      setSelected(
        citation
      );

      setViewerLoading(
        true
      );

      try {
        const detail =
          await citationService
            .context(
              citation
            );

        setSelected(
          detail
        );
      } catch (
        requestError
      ) {
        setSelected({
          ...citation,

          _detail_error:
            getErrorMessage(
              requestError,
            ),
        });
      } finally {
        setViewerLoading(
          false
        );
      }
    };


  // ========================================================
  // UI
  // ========================================================

  return (
    <div className="citations-page p-3 sm:p-5">

      <div className="mb-4 flex flex-col gap-3 border-b border-slate-800 pb-4 sm:flex-row sm:items-center">

        <div className="relative min-w-0 flex-1">

          <Search className="absolute left-3 top-3 h-4 w-4 text-slate-600" />

          <input
            className="field pl-10"
            placeholder="Search chunks by keyword, snippet text, or chunk ID..."
            value={query}
            onChange={
              (event) =>
                setQuery(
                  event.target.value
                )
            }
          />

        </div>

        <div className="rounded-lg border border-slate-800 bg-ink-900 px-3 py-2 text-[10px] text-emerald-400">
          Live PostgreSQL Citations
        </div>

      </div>


      <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">

        {filtered.map(
          (
            citation,
            index,
          ) => {

            const score =
              (
                citation.relevance_score
                != null
              )
                ? Number(
                    citation.relevance_score
                  )
                : (
                    citation.confidence_score
                    != null
                  )
                  ? Number(
                      citation.confidence_score
                    )
                  : null;

            return (
              <Card
                key={
                  citation.id
                  ||
                  citation.chunk_id
                  ||
                  index
                }
                className="citation-card p-4"
              >

                <div className="flex items-start gap-3">

                  <div
                    data-document-type={citationDocumentType(citation)}
                    className="document-card-icon-wrap citation-document-icon-tile grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-emerald-900 bg-emerald-950/30"
                  >

                    <FileText className="document-card-icon h-4 w-4 text-emerald-400" />

                  </div>


                  <div className="min-w-0 flex-1">

                    <div className="truncate text-xs font-semibold text-white">

                      {
                        citation.document_name
                        ||
                        citation.filename
                        ||
                        "Indexed source"
                      }

                    </div>


                    <div className="mt-1 truncate text-[9px] text-slate-500">

                      {
                        citation.page_number
                        != null
                          ? `Page ${citation.page_number}`
                          : "Page N/A"
                      }

                      {
                        citation.chunk_id
                          ? ` • ${citation.chunk_id}`
                          : ""
                      }

                    </div>

                  </div>


                  {
                    Number.isFinite(
                      score
                    )
                    && (
                      <span className="rounded-md border border-emerald-900/70 bg-emerald-950/40 px-2 py-1 text-[9px] font-bold text-emerald-400">

                        {
                          Math.round(
                            score * 100
                          )
                        }% Sim

                      </span>
                    )
                  }

                </div>


                <p className="mt-4 line-clamp-3 min-h-[60px] text-xs italic leading-5 text-slate-400">

                  “{
                    citation.quoted_text
                    ||
                    "No quoted_text was persisted for this citation."
                  }”

                </p>


                <div className="mt-4 flex items-center justify-between border-t border-slate-800 pt-3">

                  <span className="text-[9px] text-slate-500">
                    Vector Log Chunk
                  </span>

                  <Button
                    variant="ghost"
                    className="citation-view-context text-violet-300"
                    onClick={
                      () =>
                        void openContext(
                          citation
                        )
                    }
                  >

                    <Eye className="h-3.5 w-3.5" />

                    View Context

                  </Button>

                </div>

              </Card>
            );
          },
        )}

      </div>


      {
        loading
        && (
          <div className="py-24 text-center text-xs text-slate-500">
            Loading persisted citations…
          </div>
        )
      }


      {
        !loading
        &&
        !filtered.length
        &&
        !error
        && (
          <div className="py-24 text-center text-xs leading-6 text-slate-600">
            No persisted citations are available for this user yet.
          </div>
        )
      }


      {
        error
        && (
          <div className="mt-4 rounded-lg border border-rose-900 bg-rose-950/20 p-3 text-xs text-rose-400">
            {error}
          </div>
        )
      }


      <CitationViewer
        citation={selected}
        loading={viewerLoading}
        onClose={
          () =>
            setSelected(
              null
            )
        }
      />

    </div>
  );
}