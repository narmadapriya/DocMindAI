import React, {
  useEffect,
  useMemo,
  useState,
} from "react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  BookOpen,
  Download,
  Sparkles,
} from "lucide-react";

import {
  Button,
  Card,
  EmptyState,
} from "../components/ui";

import {
  MultiDocumentDropdown,
} from "../components/MultiDocumentDropdown";

import {
  CitationList,
  CitationViewer,
} from "../components/CitationPanel";

import {
  useDocumentStore,
} from "../store/documentStore";

import {
  useAuthStore,
} from "../store/authStore";

import {
  ragService,
} from "../services/rag";

import {
  getErrorMessage,
} from "../services/api";

import {
  documentReady,
} from "../utils/format";

import {
  downloadSummaryPdf,
} from "../utils/pdfExport";


const MIN_SUMMARY_DOCUMENTS = 1;
const MAX_SUMMARY_DOCUMENTS = 1;


export default function Summaries() {
  const user =
    useAuthStore(
      (state) =>
        state.user,
    );

  const documents =
    useDocumentStore(
      (state) =>
        state.documents,
    );

  const selected =
    useDocumentStore(
      (state) =>
        state.selectedIds,
    );

  const setSelected =
    useDocumentStore(
      (state) =>
        state.setSelected,
    );

  const load =
    useDocumentStore(
      (state) =>
        state.load,
    );

  const [
    summaryType,
    setSummaryType,
  ] = useState(
    "file",
  );

  const [
    result,
    setResult,
  ] = useState(
    null,
  );

  const [
    loading,
    setLoading,
  ] = useState(
    false,
  );

  const [
    error,
    setError,
  ] = useState(
    "",
  );

  const [
    citation,
    setCitation,
  ] = useState(
    null,
  );


  useEffect(
    () => {
      void load();
    },
    [
      load,
    ],
  );


  const readyDocuments =
    useMemo(
      () =>
        documents.filter(
          documentReady,
        ),
      [
        documents,
      ],
    );


  const selectedReadyIds =
    useMemo(
      () =>
        selected
          .filter(
            (id) =>
              readyDocuments.some(
                (document) =>
                  String(
                    document.id,
                  )
                  ===
                  String(id),
              ),
          )
          .slice(
            0,
            MAX_SUMMARY_DOCUMENTS,
          ),
      [
        readyDocuments,
        selected,
      ],
    );


  const canGenerate =
    selectedReadyIds.length
      >= MIN_SUMMARY_DOCUMENTS
    &&
    selectedReadyIds.length
      <= MAX_SUMMARY_DOCUMENTS;


  const run =
    async () => {
      if (
        !user
        ||
        !canGenerate
      ) {
        return;
      }

      setLoading(true);
      setError("");

      try {
        const data =
          await ragService.summary({
            user_id:
              user.id,

            document_ids:
              selectedReadyIds,

            top_k:
              Number(
                import.meta.env
                  .VITE_DEFAULT_TOP_K
                ||
                5,
              ),

            scope:
              summaryType === "executive"
                ? "executive"
                : "document",
          });

        setResult(data);
      } catch (
        requestError
      ) {
        setError(
          getErrorMessage(
            requestError,
          ),
        );
      } finally {
        setLoading(false);
      }
    };


  return (
    <div className="summary-page grid gap-3 p-3 sm:p-4 xl:grid-cols-[350px_minmax(0,1fr)]">

      <Card className="h-fit p-3 sm:p-4">

        <h2 className="text-[13px] font-semibold text-white">
          Summary Creator
        </h2>

        <p className="mt-1 text-[9px] leading-4 text-slate-500">
          Generate a grounded summary for 1 RAG-ready document.
        </p>


        <div className="mt-4">

          <div className="label text-[9px]">
            Summary Target Scope
          </div>

          <div className="grid grid-cols-2 gap-1.5">

            {[
              "file",
              "executive",
            ].map(
              (type) => (
                <button
                  key={type}
                  type="button"
                  onClick={
                    () =>
                      setSummaryType(
                        type,
                      )
                  }
                  className={`theme-segment h-8 rounded-md border px-2.5 text-[10px] capitalize ${
                    summaryType
                    === type
                      ? "border-violet-600 bg-violet-950/40 text-violet-200"
                      : "border-slate-700 bg-ink-800 text-slate-400"
                  }`}
                >
                  {
                    type === "file"
                      ? "File Summary"
                      : "Executive"
                  }
                </button>
              ),
            )}

          </div>

        </div>


        <div className="mt-3.5">

          <div className="label text-[9px]">
            Select Document (
            {selectedReadyIds.length}/
            {MAX_SUMMARY_DOCUMENTS})
          </div>

          <MultiDocumentDropdown
            documents={
              readyDocuments
            }
            selectedIds={
              selectedReadyIds
            }
            onChange={
              setSelected
            }
            min={
              MIN_SUMMARY_DOCUMENTS
            }
            max={
              MAX_SUMMARY_DOCUMENTS
            }
            singleSelect
            placeholder="Select 1 document."
          />

          <p className="mt-1.5 text-[8px] text-slate-500">
            Select exactly 1 RAG-ready document.
          </p>

        </div>


        {
          error
          && (
            <div className="mt-2.5 rounded-md border border-rose-900/70 bg-rose-950/20 p-2 text-[10px] text-rose-400">
              {error}
            </div>
          )
        }


        <Button
          onClick={
            () =>
              void run()
          }
          loading={loading}
          disabled={
            !canGenerate
          }
          className="mt-3 h-8 w-full text-[11px]"
        >
          <Sparkles className="h-3.5 w-3.5" />
          Generate Summary
        </Button>

      </Card>


      <Card className="min-h-[470px] overflow-hidden">

        <div className="border-b border-slate-800 px-3.5 py-2.5 text-[10px] font-semibold text-slate-300">

          <BookOpen className="mr-1.5 inline h-3.5 w-3.5 text-violet-300" />

          Document Report Sheet

        </div>


        {
          !result
          ? (
            <EmptyState
              icon={
                <Sparkles className="h-8 w-8" />
              }
              title="Summary Area Empty"
              description="Select 1 document and click Generate Summary to run the existing summarization pipeline."
            />
          )
          : (
            <div className="p-3.5 sm:p-5">

              <div className="mb-4 flex flex-col gap-2.5 sm:flex-row sm:items-center sm:justify-between">

                <div>

                  <h3 className="text-[13px] font-semibold text-white">
                    Generated Summary
                  </h3>

                  <p className="mt-1 text-[9px] text-slate-500">
                    Generated by the existing RAG summarization endpoint.
                  </p>

                </div>


                <Button
                  variant="secondary"
                  className="h-8 px-2.5 text-[10px]"
                  onClick={
                    () =>
                      downloadSummaryPdf({
                        summary:
                          result.summary
                          ||
                          result.answer
                          ||
                          "No summary returned.",

                        citations:
                          result.citations
                          ||
                          [],
                      })
                  }
                >
                  <Download className="h-3.5 w-3.5" />
                  Download as PDF
                </Button>

              </div>


              <div className="prose prose-invert max-w-none text-[12px] leading-6 text-slate-300">

                <ReactMarkdown
                  remarkPlugins={[
                    remarkGfm,
                  ]}
                >
                  {
                    result.summary
                    ||
                    result.answer
                    ||
                    "No summary returned."
                  }
                </ReactMarkdown>

              </div>


              <CitationList
                citations={
                  result.citations
                  ||
                  []
                }
                onOpen={
                  setCitation
                }
              />

            </div>
          )
        }

      </Card>


      <CitationViewer
        citation={citation}
        onClose={
          () =>
            setCitation(null)
        }
      />

    </div>
  );
}