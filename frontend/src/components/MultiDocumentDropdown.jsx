import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  Check,
  ChevronDown,
  FileSpreadsheet,
  FileText,
  Search,
  Trash2,
  X,
} from "lucide-react";

import {
  formatBytes,
  formatDate,
} from "../utils/format";


const DOCUMENT_LIST_MAX_HEIGHT = 125;


// ==========================================================
// Document Helpers
// ==========================================================

function documentName(document) {
  return (
    document.original_filename
    || document.filename
    || "Untitled document"
  );
}


function documentExtension(document) {
  const explicitType = String(
    document.file_type || "",
  )
    .replace(".", "")
    .trim()
    .toLowerCase();

  if (explicitType) {
    return explicitType;
  }

  const name = documentName(
    document,
  );

  return (
    name
      .split(".")
      .pop()
      ?.toLowerCase()
    || "document"
  );
}


function documentTypeLabel(document) {
  const extension = documentExtension(
    document,
  );

  const labels = {
    xlsx: "Excel",
    xls: "Excel",
    csv: "CSV",
    docx: "Word",
    doc: "Word",
    pdf: "PDF",
    txt: "TXT",
  };

  return (
    labels[extension]
    || extension.toUpperCase()
  );
}


function documentIconStyles(document) {
  const extension = documentExtension(
    document,
  );

  if (
    extension === "xlsx"
    || extension === "xls"
    || extension === "csv"
  ) {
    return {
      wrapper:
        "border-emerald-700/50 bg-emerald-950/35",

      icon:
        "text-emerald-400",

      spreadsheet:
        true,
    };
  }

  if (
    extension === "docx"
    || extension === "doc"
  ) {
    return {
      wrapper:
        "border-blue-700/50 bg-blue-950/35",

      icon:
        "text-blue-400",

      spreadsheet:
        false,
    };
  }

  if (
    extension === "pdf"
  ) {
    return {
      wrapper:
        "border-rose-700/50 bg-rose-950/35",

      icon:
        "text-rose-400",

      spreadsheet:
        false,
    };
  }

  return {
    wrapper:
      "border-violet-700/50 bg-violet-950/30",

    icon:
      "text-slate-500",

    spreadsheet:
      false,
  };
}


// ==========================================================
// Document Type Icon
// ==========================================================

function DocumentTypeIcon({
  document,
  className = "h-4 w-4",
}) {
  const styles =
    documentIconStyles(
      document,
    );

  const Icon =
    styles.spreadsheet
      ? FileSpreadsheet
      : FileText;

  return (
    <Icon
      className={
        `${className} ${styles.icon}`
      }
    />
  );
}


// ==========================================================
// Multi Document Dropdown
// ==========================================================

export function MultiDocumentDropdown({
  documents,
  selectedIds,
  onChange,
  min = 2,
  max = 5,
  placeholder = "Select 2 to 5 documents",
  singleSelect = false,
}) {
  const [
    open,
    setOpen,
  ] = useState(false);

  const [
    search,
    setSearch,
  ] = useState("");

  const wrapperRef =
    useRef(null);


  // ========================================================
  // Close Dropdown When Clicking Outside
  // ========================================================

  useEffect(() => {
    const close =
      (event) => {
        if (
          wrapperRef.current
          &&
          !wrapperRef.current.contains(
            event.target,
          )
        ) {
          setOpen(false);
        }
      };

    document.addEventListener(
      "mousedown",
      close,
    );

    return () =>
      document.removeEventListener(
        "mousedown",
        close,
      );
  }, []);


  // ========================================================
  // Selected Documents
  // ========================================================

  const selectedDocuments =
    useMemo(
      () =>
        documents.filter(
          (document) =>
            selectedIds.some(
              (id) =>
                String(id)
                ===
                String(
                  document.id,
                ),
            ),
        ),
      [
        documents,
        selectedIds,
      ],
    );


  // ========================================================
  // Search / Filter
  // ========================================================

  const filteredDocuments =
    useMemo(
      () => {
        const term =
          search
            .trim()
            .toLowerCase();

        if (!term) {
          return documents;
        }

        return documents.filter(
          (document) => {
            const searchable = [
              documentName(
                document,
              ),

              documentTypeLabel(
                document,
              ),

              document.file_type,
            ]
              .filter(Boolean)
              .join(" ")
              .toLowerCase();

            return searchable.includes(
              term,
            );
          },
        );
      },
      [
        documents,
        search,
      ],
    );


  // ========================================================
  // Toggle Selection
  // ========================================================

  const toggle =
    (id) => {
      const selected =
        selectedIds.some(
          (selectedId) =>
            String(
              selectedId,
            )
            ===
            String(id),
        );

      if (singleSelect) {
        onChange(
          selected
            ? []
            : [id],
        );

        return;
      }

      if (selected) {
        onChange(
          selectedIds.filter(
            (selectedId) =>
              String(
                selectedId,
              )
              !==
              String(id),
          ),
        );

        return;
      }

      if (
        selectedIds.length
        >= max
      ) {
        return;
      }

      onChange([
        ...selectedIds,
        id,
      ]);
    };


  // ========================================================
  // Clear Selection
  // ========================================================

  const clearSelection =
    () => {
      onChange([]);
    };


  // ========================================================
  // UI
  // ========================================================

  return (
    <div
      ref={wrapperRef}
      className="multi-document-dropdown relative"
    >

      {/*
        Local scrollbar styling only for this dropdown.

        - Scrollbar appears only when content exceeds max height.
        - Smooth vertical scrolling.
        - Chrome / Edge native up/down buttons are removed.
        - Only track + draggable thumb remain.
      */}

      <style>
        {`
          .docmind-document-scroll {
            overflow-y: auto;
            overflow-x: hidden;
            scroll-behavior: smooth;
            scrollbar-gutter: stable;
          }


          /* ==================================================
             Chrome / Edge
          ================================================== */

          .docmind-document-scroll::-webkit-scrollbar {
            width: 8px;
          }


          .docmind-document-scroll::-webkit-scrollbar-track {
            background: #111c33;
            border-radius: 999px;
          }


          /* ==================================================
             Remove ALL top/bottom arrow buttons
          ================================================== */

          .docmind-document-scroll::-webkit-scrollbar-button,
          .docmind-document-scroll::-webkit-scrollbar-button:single-button,
          .docmind-document-scroll::-webkit-scrollbar-button:vertical,
          .docmind-document-scroll::-webkit-scrollbar-button:vertical:decrement,
          .docmind-document-scroll::-webkit-scrollbar-button:vertical:increment,
          .docmind-document-scroll::-webkit-scrollbar-button:start:decrement,
          .docmind-document-scroll::-webkit-scrollbar-button:end:increment {
            -webkit-appearance: none !important;
            appearance: none !important;

            background: transparent !important;
            background-image: none !important;

            border: none !important;

            width: 0 !important;
            height: 0 !important;

            min-width: 0 !important;
            min-height: 0 !important;
          }


          /* ==================================================
             Keep only draggable thumb
          ================================================== */

          .docmind-document-scroll::-webkit-scrollbar-thumb {
            background: linear-gradient(
              180deg,
              #825cff 0%,
              #5a4df7 100%
            );

            border-radius: 999px;
          }


          .docmind-document-scroll::-webkit-scrollbar-thumb:hover {
            background: #8b6cff;
          }


          .docmind-document-scroll::-webkit-scrollbar-corner {
            display: none;
          }
        `}
      </style>


      {/* ====================================================
          Dropdown Trigger
      ==================================================== */}

      <button
        type="button"
        className="multi-document-trigger field flex min-h-[38px] w-full items-center justify-between gap-2 border-violet-700/70 px-2.5 text-left transition-colors focus:border-violet-500"
        onClick={
          () =>
            setOpen(
              (value) =>
                !value,
            )
        }
        aria-expanded={
          open
        }
        aria-haspopup="listbox"
      >

        <span className="flex min-w-0 flex-1 items-center gap-2">

          <span className="multi-document-trigger-icon grid h-7 w-7 shrink-0 place-items-center rounded-full bg-violet-900/45 text-violet-300">

            <FileText className="h-3.5 w-3.5" />

          </span>


          <span className="multi-document-trigger-text min-w-0 flex-1 truncate text-[10px] font-medium text-slate-200">

            {
              selectedDocuments.length
                ? `${selectedDocuments.length} ${
                    selectedDocuments.length
                    === 1
                      ? "document"
                      : "documents"
                  } selected`
                : placeholder
            }

          </span>

        </span>


        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 text-slate-300 transition-transform duration-200 ${
            open
              ? "rotate-180"
              : ""
          }`}
        />

      </button>


      {/* ====================================================
          Selected Document Chips
      ==================================================== */}

      {
        selectedDocuments.length
        > 0
        && (
          <div className="multi-document-selected-chips mt-1.5 flex flex-wrap gap-1.5">

            {
              selectedDocuments.map(
                (document) => (
                  <span
                    key={
                      document.id
                    }
                    data-document-type={documentExtension(document).toUpperCase()}
                    className="multi-document-chip inline-flex max-w-full items-center gap-1 rounded-md border border-violet-700 bg-violet-950/35 px-1.5 py-0.5 text-[9px] text-violet-200"
                  >

                    <DocumentTypeIcon
                      document={document}
                      className="document-card-icon h-2.5 w-2.5 shrink-0"
                    />


                    <span className="multi-document-chip-name max-w-[190px] truncate">

                      {
                        documentName(
                          document,
                        )
                      }

                    </span>


                    <button
                      type="button"
                      onClick={
                        () =>
                          toggle(
                            document.id,
                          )
                      }
                      title="Remove document"
                      className="multi-document-chip-remove rounded p-0.5 hover:bg-violet-900/60"
                    >

                      <X className="h-2.5 w-2.5" />

                    </button>

                  </span>
                ),
              )
            }

          </div>
        )
      }


      {/* ====================================================
          Dropdown
      ==================================================== */}

      {
        open
        && (
          <div
            className="multi-document-menu theme-dropdown absolute left-0 right-0 z-50 mt-1.5 overflow-hidden rounded-lg border border-slate-700/90 bg-[#081126] shadow-2xl"
            role="listbox"
            aria-multiselectable={!singleSelect}
          >

            {/* ==============================================
                Search
            ============================================== */}

            <div className="multi-document-search-row border-b border-slate-800/80 p-2">

              <div className="relative">

                <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-500" />


                <input
                  type="search"
                  value={
                    search
                  }
                  onChange={
                    (event) =>
                      setSearch(
                        event.target.value,
                      )
                  }
                  placeholder="Search documents..."
                  className="multi-document-search field h-8 w-full pl-8 pr-2.5 text-[10px]"
                  onClick={
                    (event) =>
                      event.stopPropagation()
                  }
                />

              </div>

            </div>


            {/* ==============================================
                Scrollable Document List
            ============================================== */}

            <div
              className="multi-document-list docmind-document-scroll px-2 py-1.5"
              style={{
                maxHeight:
                  `${DOCUMENT_LIST_MAX_HEIGHT}px`,
              }}
            >

              {
                !filteredDocuments.length
                ? (
                  <div className="grid min-h-[72px] place-items-center px-3 text-center text-[9px] text-slate-500">

                    {
                      documents.length
                        ? "No documents match your search."
                        : "No RAG-ready documents available."
                    }

                  </div>
                )
                : (
                  <div className="space-y-1">

                    {
                      filteredDocuments.map(
                        (document) => {

                          const selected =
                            selectedIds.some(
                              (id) =>
                                String(id)
                                ===
                                String(
                                  document.id,
                                ),
                            );


                          const disabled =
                            !singleSelect
                            &&
                            !selected
                            &&
                            selectedIds.length
                            >= max;


                          const iconStyles =
                            documentIconStyles(
                              document,
                            );


                          const updated =
                            document.updated_at
                            ||
                            document.created_at;


                          const metadata = [
                            documentTypeLabel(
                              document,
                            ),

                            Number(
                              document.file_size
                              || 0,
                            ) > 0
                              ? formatBytes(
                                  document.file_size,
                                )
                              : null,

                            updated
                              ? `Updated ${formatDate(updated)}`
                              : null,
                          ]
                            .filter(Boolean)
                            .join(" • ");


                          return (
                            <button
                              key={
                                document.id
                              }
                              type="button"
                              role="option"
                              aria-selected={
                                selected
                              }
                              disabled={
                                disabled
                              }
                              onClick={
                                () =>
                                  toggle(
                                    document.id,
                                  )
                              }
                              data-document-type={documentExtension(document).toUpperCase()}
                              className={`multi-document-option document-card-ui flex w-full items-center gap-2 rounded-lg border px-2.5 py-2 text-left transition-colors ${
                                selected
                                  ? "border-violet-500 bg-violet-950/45 shadow-[inset_0_0_18px_rgba(109,40,217,0.10)]"
                                  : "border-slate-800 bg-slate-950/20 hover:border-slate-700 hover:bg-slate-900/55"
                              } ${
                                disabled
                                  ? "cursor-not-allowed opacity-40"
                                  : ""
                              }`}
                            >

                              {/* Document Icon */}

                              <span
                                className={`document-card-icon-wrap grid h-8 w-8 shrink-0 place-items-center rounded-md border ${iconStyles.wrapper}`}
                              >

                                <DocumentTypeIcon
                                  document={
                                    document
                                  }
                                  className="document-card-icon h-4 w-4"
                                />

                              </span>


                              {/* Document Information */}

                              <span className="document-card-copy min-w-0 flex-1">

                                <span className="document-card-title block truncate text-[10px] font-semibold text-slate-100">

                                  {
                                    documentName(
                                      document,
                                    )
                                  }

                                </span>


                                <span className="document-card-meta mt-0.5 block truncate text-[8px] text-slate-500">

                                  {
                                    metadata
                                    ||
                                    "Repository document"
                                  }

                                </span>

                              </span>


                              {/* Selected Indicator */}

                              <span
                                className={`document-card-select grid h-5 w-5 shrink-0 place-items-center rounded-full border transition-colors ${
                                  selected
                                    ? "border-violet-500 bg-violet-600 text-white"
                                    : "border-slate-600 bg-transparent text-transparent"
                                }`}
                              >

                                <Check className="h-3 w-3" />

                              </span>

                            </button>
                          );
                        },
                      )
                    }

                  </div>
                )
              }

            </div>


            {/* ==============================================
                Footer
            ============================================== */}

            <div className="multi-document-footer flex items-center justify-between gap-2 border-t border-slate-800 bg-[#0a1327] px-2.5 py-2">

              <span className="multi-document-summary text-[9px] text-slate-500">

                <strong className="font-semibold text-violet-400">

                  {
                    selectedDocuments.length
                  }

                </strong>

                {" "}
                of
                {" "}

                <strong className="font-semibold text-slate-300">

                  {
                    max
                  }

                </strong>

                {" "}
                {
                  max === 1
                    ? "document selected"
                    : "documents selected"
                }

              </span>


              <button
                type="button"
                onClick={
                  clearSelection
                }
                disabled={
                  !selectedIds.length
                }
                className="multi-document-clear inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[9px] font-medium text-violet-400 transition-colors hover:bg-violet-950/40 disabled:cursor-not-allowed disabled:opacity-40"
              >

                <Trash2 className="h-3 w-3" />

                Clear selection

              </button>

            </div>

          </div>
        )
      }

    </div>
  );
} 