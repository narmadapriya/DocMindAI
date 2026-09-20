import React, { useState } from "react";

import {
  AlertCircle,
  CheckCircle2,
  FileText,
  FileUp,
  LoaderCircle,
  XCircle,
} from "lucide-react";

import { useDropzone } from "react-dropzone";

import {
  MAX_UPLOAD_FILES,
  useDocumentStore,
} from "../store/documentStore";

const ACCEPT = {
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
    ".docx",
  ],
  "text/plain": [".txt"],
  "text/csv": [".csv"],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
    ".xlsx",
  ],
};

function StatusIcon({ status }) {
  if (status === "uploading") {
    return (
      <LoaderCircle className="upload-status-icon h-3.5 w-3.5 animate-spin text-violet-400" />
    );
  }

  if (status === "ready") {
    return (
      <CheckCircle2 className="upload-status-icon h-3.5 w-3.5 text-emerald-400" />
    );
  }

  if (status === "failed") {
    return (
      <XCircle className="upload-status-icon h-3.5 w-3.5 text-rose-400" />
    );
  }

  return null;
}

export function UploadDropzone() {
  const uploadFiles = useDocumentStore(
    (state) => state.uploadFiles,
  );

  const uploads = useDocumentStore(
    (state) => state.uploads,
  );

  const storeError = useDocumentStore(
    (state) => state.error,
  );

  const [localError, setLocalError] = useState("");

  const {
    getRootProps,
    getInputProps,
    isDragActive,
  } = useDropzone({
    multiple: true,
    maxFiles: MAX_UPLOAD_FILES,
    accept: ACCEPT,

    onDropAccepted: (files) => {
      setLocalError("");
      void uploadFiles(files);
    },

    onDropRejected: (items) => {
      const tooMany = items.some((rejection) =>
        rejection.errors.some(
          (error) =>
            error.code === "too-many-files",
        ),
      );

      if (tooMany) {
        setLocalError(
          `Maximum ${MAX_UPLOAD_FILES} files can be uploaded at a time.`,
        );

        return;
      }

      setLocalError(
        items
          .map(
            (rejection) =>
              `${rejection.file.name}: ${rejection.errors
                .map((error) => error.message)
                .join(", ")}`,
          )
          .join(" | "),
      );
    },
  });

  const queue = Object.values(uploads);

  return (
    <div>
      <div
        {...getRootProps()}
        className={`grid min-h-[135px] cursor-pointer place-items-center rounded-xl border border-dashed p-5 text-center transition ${
          isDragActive
            ? "border-violet-500 bg-violet-950/25"
            : "border-slate-700 bg-[#050b1c] hover:border-violet-700"
        }`}
      >
        <input {...getInputProps()} />

        <div>
          <div className="mx-auto mb-3 grid h-10 w-10 place-items-center rounded-full border border-violet-800 bg-violet-950/50 text-violet-300">
            <FileUp className="h-5 w-5" />
          </div>

          <div className="text-xs font-semibold text-slate-200">
            Drag & drop files here, or browse files
          </div>

          <div className="mt-1 text-[9px] text-slate-500">
            PDF, DOCX, TXT, CSV, XLSX • Maximum{" "}
            {MAX_UPLOAD_FILES} files per batch • Maximum 25 MB
            per file
          </div>
        </div>
      </div>

      {(localError || storeError) && (
        <div className="mt-3 flex gap-2 rounded-lg border border-rose-900 bg-rose-950/20 p-2.5 text-[10px] text-rose-300">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />

          <span>
            {localError || storeError}
          </span>
        </div>
      )}

      {!!queue.length && (
        <div className="mt-3 space-y-2">
          {queue.map((upload) => (
            <div
              key={upload.key || upload.name}
              className="document-card upload-document-card rounded-lg border border-slate-800 bg-ink-800 p-2.5"
            >
              <div className="flex items-center gap-2 text-xs">
                <FileText
                  data-document-type={String(
                    upload.name || "",
                  )
                    .split(".")
                    .pop()
                    ?.toUpperCase()}
                  className="document-card-file-icon upload-document-file-icon hidden h-4 w-4 shrink-0"
                />

                <StatusIcon
                  status={upload.status}
                />

                <span className="document-card-title min-w-0 flex-1 truncate text-slate-300">
                  {upload.name}
                </span>

                <span className="document-card-meta text-[10px] capitalize text-slate-500">
                  {upload.status}
                </span>
              </div>

              <div className="mt-2 h-1 overflow-hidden rounded-full bg-slate-800">
                <div
                  className={`h-full transition-all ${
                    upload.status === "failed"
                      ? "bg-rose-500"
                      : upload.status === "ready"
                        ? "bg-emerald-500"
                        : "bg-violet-500"
                  }`}
                  style={{
                    width: `${upload.progress}%`,
                  }}
                />
              </div>

              {upload.error && (
                <p className="mt-1 text-[10px] text-rose-400">
                  {upload.error}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}