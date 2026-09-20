import React, {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  ArrowDownRight,
  ArrowLeftRight,
  ArrowUpRight,
  Download,
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
  downloadComparisonPdf,
} from "../utils/pdfExport";

const MAX_COMPARE_DOCUMENTS = 5;

function parseComparableNumber(value) {
  if (
    value == null ||
    value === "N/A"
  ) {
    return null;
  }

  const text =
    String(value)
      .trim()
      .replace(/,/g, "");

  const multiplier =
    /\b(?:b|bn|billion)\b/i.test(
      text,
    )
      ? 1_000_000_000
      : /\b(?:m|million)\b/i.test(
            text,
          )
        ? 1_000_000
        : /\b(?:k|thousand)\b/i.test(
              text,
            )
          ? 1_000
          : 1;

  const match =
    text.match(
      /[-+]?\d+(?:\.\d+)?/,
    );

  if (!match) {
    return null;
  }

  const numeric =
    Number(match[0]);

  return Number.isFinite(
    numeric,
  )
    ? numeric * multiplier
    : null;
}

function percentChange(
  first,
  second,
) {
  const a =
    parseComparableNumber(
      first,
    );

  const b =
    parseComparableNumber(
      second,
    );

  if (
    a == null ||
    b == null ||
    a === 0
  ) {
    return null;
  }

  return (
    ((b - a) /
      Math.abs(a)) *
    100
  );
}

export default function Comparisons() {
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
    metric,
    setMetric,
  ] = useState(
    "Revenue",
  );

  const [
    result,
    setResult,
  ] = useState(null);

  const [
    loading,
    setLoading,
  ] = useState(false);

  const [
    error,
    setError,
  ] = useState("");

  const [
    citation,
    setCitation,
  ] = useState(null);

  useEffect(() => {
    void load();
  }, [load]);

  const readyDocs =
    documents.filter(
      documentReady,
    );

  const selectedDocs =
    readyDocs.filter(
      (document) =>
        selected.some(
          (id) =>
            String(id) ===
            String(
              document.id,
            ),
        ),
    );

  const clearComparison =
    () => {
      setSelected([]);
      setMetric("");
      setResult(null);
      setError("");
      setCitation(null);
    };


  const run =
    async () => {
      if (
        !user ||
        selected.length < 2 ||
        selected.length >
          MAX_COMPARE_DOCUMENTS ||
        !metric.trim()
      ) {
        return;
      }

      const metrics =
        metric
          .split(",")
          .map(
            (value) =>
              value.trim(),
          )
          .filter(Boolean);

      if (!metrics.length) {
        return;
      }

      setLoading(true);
      setError("");

      try {
        const response =
          await ragService.compare({
            user_id:
              user.id,
            document_ids:
              selected,
            metrics,
            top_k: 10,
          });

        setResult(
          response,
        );
      } catch (requestError) {
        setError(
          getErrorMessage(
            requestError,
          ),
        );
      } finally {
        setLoading(
          false,
        );
      }
    };

  const rows =
    result?.comparison ||
    [];

  const documentNames =
    selectedDocs.map(
      (document) =>
        document.original_filename ||
        document.filename,
    );

  return (
    <div
      className="
        space-y-4
        p-3
        sm:p-5
      "
    >
      <Card
        className="
          p-4
          sm:p-5
        "
      >
        <div
          className="
            mb-3
          "
        >
          <h2
            className="
              text-base
              font-bold
              text-white
            "
          >
            AI Comparison Setup
          </h2>

          <p
            className="
              mt-1
              text-[10px]
              text-slate-500
            "
          >
            Select 2 to 5
            documents, enter one
            or more metrics, and
            run the frozen
            comparison pipeline.
          </p>
        </div>

        <div
          className="
            grid
            gap-4
            xl:grid-cols-[1.1fr_.85fr_300px]
            xl:items-start
          "
        >
          <div>
            <div
              className="label"
            >
              Select Documents
              {" "}
              ({selected.length}/
              {MAX_COMPARE_DOCUMENTS})
            </div>

            <MultiDocumentDropdown
              documents={
                readyDocs
              }
              selectedIds={
                selected
              }
              onChange={
                setSelected
              }
              min={2}
              max={
                MAX_COMPARE_DOCUMENTS
              }
              placeholder="Select 2 to 5 documents"
            />
          </div>

          <div>
            <label
              className="label"
            >
              Analysis Matrix
              Focus
            </label>

            <input
              className="field h-8 text-[12px]"
              value={metric}
              onChange={(
                event,
              ) =>
                setMetric(
                  event.target
                    .value,
                )
              }
              placeholder="Revenue, Units Sold"
            />

            <p
              className="
                mt-1
                text-[9px]
                text-slate-600
              "
            >
              Separate multiple
              metrics with commas.
            </p>
          </div>

          <div
            className="
              xl:pt-[20px]
            "
          >
            <div className="flex gap-2">
              <Button
                className="flex-1"
                disabled={
                  selected.length <
                    2 ||
                  selected.length >
                    MAX_COMPARE_DOCUMENTS ||
                  !metric.trim()
                }
                loading={loading}
                onClick={() =>
                  void run()
                }
              >
                <ArrowLeftRight
                  className="
                    h-4
                    w-4
                  "
                />
                Run Comparison
              </Button>

              <Button
                type="button"
                variant="secondary"
                disabled={loading}
                onClick={clearComparison}
                className="px-3 text-[11px]"
              >
                Clear
              </Button>
            </div>
          </div>
        </div>

        {error && (
          <div
            className="
              mt-4
              rounded-lg
              border
              border-rose-900
              bg-rose-950/20
              p-3
              text-xs
              text-rose-400
            "
          >
            {error}
          </div>
        )}
      </Card>

      <Card
        className="
          min-h-[400px]
          overflow-hidden
        "
      >
        {!result ? (
          <EmptyState
            icon={
              <ArrowLeftRight
                className="
                  h-10
                  w-10
                "
              />
            }
            title="Comparison Results"
            description="Select 2 or more documents and click Run Comparison."
          />
        ) : (
          <div
            className="
              p-4
              sm:p-5
            "
          >
            <div
              className="
                mb-3
                flex
                flex-col
                gap-3
                sm:flex-row
                sm:items-center
                sm:justify-between
              "
            >
              <div>
                <h3
                  className="
                    text-base
                    font-bold
                    text-white
                  "
                >
                  Comparison Results
                </h3>

                <p
                  className="
                    mt-1
                    text-[10px]
                    text-slate-500
                  "
                >
                  Comparison of key
                  metrics across the
                  selected documents.
                </p>
              </div>

              <Button
                variant="secondary"
                onClick={() =>
                  downloadComparisonPdf({
                    documentNames,
                    rows,
                    answer:
                      result.answer ||
                      "",
                    citations:
                      result.citations ||
                      [],
                  })
                }
              >
                <Download
                  className="
                    h-4
                    w-4
                  "
                />
                Download as PDF
              </Button>
            </div>

            <div
              className="
                table-scroll
                overflow-x-auto
                rounded-xl
                border
                border-slate-800
              "
            >
              <table
                className="
                  min-w-[900px]
                  w-full
                  border-collapse
                  text-left
                "
              >
                <thead
                  className="
                    bg-slate-800/70
                    text-[10px]
                    uppercase
                    tracking-wide
                    text-slate-400
                  "
                >
                  <tr>
                    <th
                      className="
                        px-3
                        py-3
                      "
                    >
                      Metric
                    </th>

                    <th
                      className="
                        px-3
                        py-3
                      "
                    >
                      {documentNames[0] ||
                        "Document A"}
                      <div
                        className="
                          text-[9px]
                          font-normal
                          text-slate-600
                        "
                      >
                        Document A
                      </div>
                    </th>

                    <th
                      className="
                        px-3
                        py-3
                      "
                    >
                      {documentNames[1] ||
                        "Document B"}
                      <div
                        className="
                          text-[9px]
                          font-normal
                          text-slate-600
                        "
                      >
                        Document B
                      </div>
                    </th>

                    <th
                      className="
                        px-3
                        py-3
                      "
                    >
                      Difference
                      <div
                        className="
                          text-[9px]
                          font-normal
                          text-slate-600
                        "
                      >
                        Backend change
                      </div>
                    </th>

                    <th
                      className="
                        px-3
                        py-3
                      "
                    >
                      Change (%)
                      <div
                        className="
                          text-[9px]
                          font-normal
                          text-slate-600
                        "
                      >
                        A vs B
                      </div>
                    </th>
                  </tr>
                </thead>

                <tbody
                  className="
                    divide-y
                    divide-slate-800
                  "
                >
                  {rows.map(
                    (
                      row,
                      index,
                    ) => {
                      const pct =
                        percentChange(
                          row.document_a,
                          row.document_b,
                        );

                      return (
                        <tr
                          key={`${row.metric}-${index}`}
                          className="
                            bg-ink-900/30
                            text-[11px]
                          "
                        >
                          <td
                            className="
                              px-3
                              py-3
                              font-semibold
                              text-slate-200
                            "
                          >
                            {row.metric}
                          </td>

                          <td
                            className="
                              px-3
                              py-3
                              text-slate-300
                            "
                          >
                            {row.document_a}
                          </td>

                          <td
                            className="
                              px-3
                              py-3
                              text-slate-300
                            "
                          >
                            {row.document_b}
                          </td>

                          <td
                            className={`
                              px-3
                              py-3
                              font-semibold
                              ${
                                String(
                                  row.change,
                                ).startsWith(
                                  "-",
                                )
                                  ? "text-rose-400"
                                  : String(
                                        row.change,
                                      ).startsWith(
                                        "+",
                                      )
                                    ? "text-emerald-400"
                                    : "text-slate-300"
                              }
                            `}
                          >
                            {row.change}
                          </td>

                          <td
                            className="
                              px-3
                              py-3
                            "
                          >
                            {pct == null ? (
                              <span
                                className="
                                  text-slate-500
                                "
                              >
                                N/A
                              </span>
                            ) : (
                              <span
                                className={`
                                  inline-flex
                                  items-center
                                  gap-1
                                  font-semibold
                                  ${
                                    pct < 0
                                      ? "text-rose-400"
                                      : pct > 0
                                        ? "text-emerald-400"
                                        : "text-slate-400"
                                  }
                                `}
                              >
                                {pct >
                                0 ? (
                                  <ArrowUpRight
                                    className="
                                      h-3
                                      w-3
                                    "
                                  />
                                ) : pct <
                                  0 ? (
                                  <ArrowDownRight
                                    className="
                                      h-3
                                      w-3
                                    "
                                  />
                                ) : null}

                                {pct >
                                0
                                  ? "+"
                                  : ""}
                                {pct.toFixed(
                                  2,
                                )}
                                %
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    },
                  )}
                </tbody>
              </table>
            </div>

            <CitationList
              citations={
                result.citations ||
                []
              }
              onOpen={
                setCitation
              }
            />
          </div>
        )}
      </Card>

      <CitationViewer
        citation={citation}
        onClose={() =>
          setCitation(null)
        }
      />
    </div>
  );
}
