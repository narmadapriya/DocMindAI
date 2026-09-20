import { create } from "zustand";

import {
  documentService,
} from "../services/documents";

import {
  getErrorMessage,
} from "../services/api";

export const MAX_UPLOAD_FILES = 10;

function uploadKey(file, index) {
  return `${file.name}-${file.size}-${file.lastModified}-${index}`;
}

function normalizeFilename(filename) {
  return String(filename || "")
    .trim()
    .toLowerCase();
}

function documentFilename(document) {
  return (
    document?.original_filename ||
    document?.filename ||
    document?.name ||
    ""
  );
}

function duplicateMessage() {
  return "The filename already present.";
}

function notify(message) {
  if (
    typeof window !== "undefined" &&
    typeof window.alert === "function"
  ) {
    window.alert(message);
  }
}

const DOCUMENT_STATUS_POLL_INTERVAL_MS = 1250;
const DOCUMENT_STATUS_POLL_MAX_ATTEMPTS = 72;

const activeStatusPolls = new Map();

function sleep(milliseconds) {
  return new Promise(
    (resolve) =>
      setTimeout(
        resolve,
        milliseconds,
      ),
  );
}

function isReadyStatus(status) {
  return (
    status?.ready_for_rag === true ||
    String(
      status?.status || "",
    ).toLowerCase() === "ready"
  );
}

function isFailedStatus(status) {
  return (
    String(
      status?.status || "",
    ).toLowerCase() === "failed"
  );
}

function mergeDocumentStatus(
  document,
  status,
) {
  return {
    ...document,
    ...status,

    // Preserve the canonical document id even if the status
    // endpoint returns only document_id.
    id:
      document?.id ||
      status?.id ||
      status?.document_id,

    status:
      status?.status ||
      document?.status,

    chunk_count:
      status?.chunk_count ??
      document?.chunk_count,

    vector_count:
      status?.vector_count ??
      document?.vector_count,

    page_count:
      status?.page_count ??
      document?.page_count,

    ready_for_rag:
      status?.ready_for_rag ??
      document?.ready_for_rag,

    processing_error:
      status?.processing_error ??
      document?.processing_error,
  };
}

async function pollDocumentUntilReady(
  documentId,
  set,
  {
    uploadEntryKey = null,
    intervalMs =
      DOCUMENT_STATUS_POLL_INTERVAL_MS,
    maxAttempts =
      DOCUMENT_STATUS_POLL_MAX_ATTEMPTS,
  } = {},
) {
  const pollKey =
    String(documentId);

  // Re-use an in-flight poll for the same document. This
  // prevents upload/reindex/UI actions from starting several
  // status loops against ChromaDB at the same time.
  if (
    activeStatusPolls.has(
      pollKey,
    )
  ) {
    return activeStatusPolls.get(
      pollKey,
    );
  }

  const pollPromise =
    (async () => {
      for (
        let attempt = 0;
        attempt < maxAttempts;
        attempt += 1
      ) {
        // Check immediately once, then poll at a short interval.
        // A 1.25-second interval keeps the UI responsive without
        // aggressively hammering PostgreSQL/ChromaDB.
        if (attempt > 0) {
          await sleep(
            intervalMs,
          );
        }

        try {
          const status =
            await documentService.status(
              documentId,
            );

          set(
            (state) => ({
              documents:
                state.documents.map(
                  (document) =>
                    String(
                      document.id,
                    ) ===
                    String(
                      documentId,
                    )
                      ? mergeDocumentStatus(
                          document,
                          status,
                        )
                      : document,
                ),

              uploads:
                uploadEntryKey
                  ? {
                      ...state.uploads,

                      [uploadEntryKey]: {
                        ...state.uploads[
                          uploadEntryKey
                        ],

                        progress: 100,

                        status:
                          isReadyStatus(
                            status,
                          )
                            ? "ready"
                            : isFailedStatus(
                                status,
                              )
                              ? "failed"
                              : "processing",

                        error:
                          isFailedStatus(
                            status,
                          )
                            ? (
                                status?.processing_error ||
                                "Document processing failed."
                              )
                            : undefined,
                      },
                    }
                  : state.uploads,
            }),
          );

          if (
            isReadyStatus(
              status,
            )
          ) {
            return status;
          }

          if (
            isFailedStatus(
              status,
            )
          ) {
            const message = (
              status?.processing_error ||
              "Document processing failed."
            );

            set({
              error: message,
            });

            return status;
          }
        } catch (error) {
          // The upload itself has already succeeded. A temporary
          // status read failure must not mark the document failed.
          // The next polling attempt can recover automatically.
          if (
            typeof console !==
              "undefined" &&
            typeof console.warn ===
              "function"
          ) {
            console.warn(
              "Document status polling failed:",
              error,
            );
          }
        }
      }

      return null;
    })();

  activeStatusPolls.set(
    pollKey,
    pollPromise,
  );

  try {
    return await pollPromise;
  } finally {
    activeStatusPolls.delete(
      pollKey,
    );
  }
}

export const useDocumentStore = create((set, get) => ({
  documents: [],
  selectedIds: [],
  loading: false,
  error: null,
  uploads: {},

  load: async () => {
    set({
      loading: true,
      error: null,
    });

    try {
      const documents =
        await documentService.list();

      set({
        documents,
      });

      // Resume polling for documents that were already processing
      // when this page/store was loaded. This is important after a
      // browser refresh or navigation because an in-memory poll from
      // the original upload no longer exists.
      documents
        .filter(
          (document) =>
            document?.id &&
            !isReadyStatus(document) &&
            !isFailedStatus(document),
        )
        .forEach((document) => {
          void pollDocumentUntilReady(
            document.id,
            set,
          );
        });

      return documents;
    } catch (error) {
      set({
        error:
          getErrorMessage(error),
      });

      return [];
    } finally {
      set({
        loading: false,
      });
    }
  },

  toggle: (id) => {
    const current =
      get().selectedIds;

    const exists =
      current.some(
        (value) =>
          String(value) ===
          String(id),
      );

    set({
      selectedIds:
        exists
          ? current.filter(
              (value) =>
                String(value) !==
                String(id),
            )
          : [
              ...current,
              id,
            ],
    });
  },

  selectOne: (id) =>
    set({
      selectedIds:
        id ? [id] : [],
    }),

  setSelected: (ids) =>
    set({
      selectedIds:
        Array.from(
          new Set(
            ids || [],
          ),
        ),
    }),

  clearSelected: () =>
    set({
      selectedIds: [],
    }),

  clearUploads: () =>
    set({
      uploads: {},
      error: null,
    }),

  uploadFiles: async (files) => {
    const batch =
      Array.from(
        files || [],
      );

    if (!batch.length) {
      return;
    }

    if (
      batch.length >
      MAX_UPLOAD_FILES
    ) {
      const message =
        `Maximum ${MAX_UPLOAD_FILES} files can be uploaded at a time.`;

      set({
        error: message,
      });

      notify(message);
      return;
    }

    const accepted =
      new Set([
        ".pdf",
        ".docx",
        ".txt",
        ".csv",
        ".xlsx",
      ]);

    const invalid =
      batch.filter(
        (file) => {
          const dot =
            file.name.lastIndexOf(
              ".",
            );

          const extension =
            dot >= 0
              ? file.name
                  .slice(dot)
                  .toLowerCase()
              : "";

          return !accepted.has(
            extension,
          );
        },
      );

    if (invalid.length) {
      const message =
        `Unsupported file type: ${invalid
          .map(
            (file) =>
              file.name,
          )
          .join(", ")}`;

      set({
        error: message,
      });

      notify(message);
      return;
    }

    // Always query the live repository before starting an upload.
    // This avoids duplicate detection against stale Zustand state.
    let repositoryDocuments;

    try {
      repositoryDocuments =
        await documentService.list();

      set({
        documents:
          repositoryDocuments,
      });
    } catch (error) {
      set({
        error:
          getErrorMessage(error),
      });
      return;
    }

    const existingNames =
      new Set(
        repositoryDocuments.map(
          (document) =>
            normalizeFilename(
              documentFilename(
                document,
              ),
            ),
        ),
      );

    const duplicateExisting =
      batch.find(
        (file) =>
          existingNames.has(
            normalizeFilename(
              file.name,
            ),
          ),
      );

    if (duplicateExisting) {
      const message =
        duplicateMessage();

      set({
        error: message,
      });

      notify(message);
      return;
    }

    // Also reject duplicate names selected in the same upload batch.
    const selectedNames =
      new Set();

    for (const file of batch) {
      const normalized =
        normalizeFilename(
          file.name,
        );

      if (
        selectedNames.has(
          normalized,
        )
      ) {
        const message =
          duplicateMessage();

        set({
          error: message,
        });

        notify(message);
        return;
      }

      selectedNames.add(
        normalized,
      );
    }

    const queued = {};

    batch.forEach(
      (file, index) => {
        const key =
          uploadKey(
            file,
            index,
          );

        queued[key] = {
          key,
          name: file.name,
          progress: 0,
          status: "queued",
        };
      },
    );

    set({
      uploads: queued,
      error: null,
    });

    // Sequential ingestion is intentionally retained because each upload
    // can trigger parsing, chunking, embedding, vector indexing and DB writes.
    for (
      let index = 0;
      index < batch.length;
      index += 1
    ) {
      const file =
        batch[index];

      const key =
        uploadKey(
          file,
          index,
        );

      set(
        (state) => ({
          uploads: {
            ...state.uploads,

            [key]: {
              ...state.uploads[
                key
              ],
              progress: 1,
              status:
                "uploading",
              error:
                undefined,
            },
          },
        }),
      );

      try {
        // Re-check the live repository immediately before each file,
        // preventing a race with another browser/session upload.
        const latestDocuments =
          await documentService.list();

        const latestNames =
          new Set(
            latestDocuments.map(
              (document) =>
                normalizeFilename(
                  documentFilename(
                    document,
                  ),
                ),
            ),
          );

        if (
          latestNames.has(
            normalizeFilename(
              file.name,
            ),
          )
        ) {
          const message =
            duplicateMessage();

          set(
            (state) => ({
              uploads: {
                ...state.uploads,

                [key]: {
                  ...state.uploads[
                    key
                  ],
                  progress: 0,
                  status: "failed",
                  error: message,
                },
              },

              error: message,
            }),
          );

          notify(message);
          continue;
        }

        const created =
          await documentService.upload(
            file,
            (progress) => {
              set(
                (state) => ({
                  uploads: {
                    ...state.uploads,

                    [key]: {
                      ...state.uploads[
                        key
                      ],
                      progress,
                      status:
                        "uploading",
                    },
                  },
                }),
              );
            },
          );

        if (!created?.id) {
          throw new Error(
            "Upload completed but the backend did not return a document id.",
          );
        }

        set(
          (state) => ({
            documents: [
              created,

              ...state.documents.filter(
                (document) =>
                  String(
                    document.id,
                  ) !==
                  String(
                    created.id,
                  ),
              ),
            ],

            uploads: {
              ...state.uploads,

              [key]: {
                ...state.uploads[
                  key
                ],
                progress: 100,
                status: "processing",
              },
            },
          }),
        );

        // Upload completion only means the file/Document row was
        // accepted. The existing backend finishes parsing,
        // chunking, embedding and ChromaDB indexing in a
        // background task. Poll without blocking the next upload
        // so this row becomes Ready as soon as the backend's
        // existing status endpoint reports ready_for_rag.
        void pollDocumentUntilReady(
          created.id,
          set,
          {
            uploadEntryKey: key,
          },
        );
      } catch (error) {
        set(
          (state) => ({
            uploads: {
              ...state.uploads,

              [key]: {
                ...state.uploads[
                  key
                ],
                status: "failed",
                error:
                  getErrorMessage(
                    error,
                  ),
              },
            },
          }),
        );
      }
    }

    await get().load();
  },

  reindex: async (id) => {
    set({
      error: null,
    });

    try {
      await documentService.reindex(
        id,
      );

      set(
        (state) => ({
          documents:
            state.documents.map(
              (document) =>
                String(
                  document.id,
                ) ===
                String(id)
                  ? {
                      ...document,
                      status:
                        "processing",
                      ready_for_rag:
                        false,
                    }
                  : document,
            ),
        }),
      );

      await pollDocumentUntilReady(
        id,
        set,
      );

      await get().load();
    } catch (error) {
      set({
        error:
          getErrorMessage(error),
      });

      throw error;
    }
  },

  remove: async (id) => {
    try {
      await documentService.remove(
        id,
      );

      set(
        (state) => ({
          documents:
            state.documents.filter(
              (document) =>
                String(
                  document.id,
                ) !==
                String(id),
            ),

          selectedIds:
            state.selectedIds.filter(
              (value) =>
                String(value) !==
                String(id),
            ),
        }),
      );
    } catch (error) {
      set({
        error:
          getErrorMessage(error),
      });

      throw error;
    }
  },
}));