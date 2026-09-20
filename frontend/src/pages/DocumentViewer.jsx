import React, { useEffect, useState } from "react";
import { Download, FileText } from "lucide-react";
import { useParams } from "react-router-dom";
import { Card, Button, StatusPill } from "../components/ui";
import { api, getErrorMessage } from "../services/api";
import { documentService } from "../services/documents";
import { formatBytes, formatDate } from "../utils/format";

export default function DocumentViewer() {
  const { id = "" } = useParams();
  const [document, setDocument] = useState(null);
  const [error, setError] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    let objectUrl = "";

    void documentService
      .get(id)
      .then(async (metadata) => {
        setDocument(metadata);

        const isPdf = (metadata.file_type || metadata.filename || "")
          .toLowerCase()
          .includes("pdf");

        if (isPdf) {
          const response = await api.get(
            `/api/upload/download/${id}`,
            { responseType: "blob" },
          );
          objectUrl = URL.createObjectURL(response.data);
          setPreviewUrl(objectUrl);
        }
      })
      .catch((requestError) => {
        setError(getErrorMessage(requestError));
      });

    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [id]);

  const download = async () => {
    if (!document) return;
    setDownloading(true);

    try {
      const response = await api.get(
        `/api/upload/download/${document.id}`,
        { responseType: "blob" },
      );

      const url = URL.createObjectURL(response.data);
      const anchor = window.document.createElement("a");
      anchor.href = url;
      anchor.download =
        document.original_filename ||
        document.filename ||
        "document";
      window.document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (requestError) {
      setError(getErrorMessage(requestError));
    } finally {
      setDownloading(false);
    }
  };

  if (error) {
    return <div className="p-5 text-sm text-rose-400">{error}</div>;
  }

  if (!document) {
    return <div className="p-5 text-sm text-slate-500">Loading document…</div>;
  }

  const isPdf = (document.file_type || document.filename || "")
    .toLowerCase()
    .includes("pdf");
  const documentType = String(
    document.file_type ||
      (document.original_filename || document.filename || "document").split(".").pop() ||
      "document",
  )
    .replace(".", "")
    .toUpperCase();

  return (
    <div className="space-y-4 p-3 sm:p-5">
      <Card
        data-document-type={documentType}
        className="document-card-ui document-viewer-file-card flex flex-wrap items-center gap-4 p-4"
      >
        <div className="document-card-icon-wrap grid h-10 w-10 place-items-center rounded-lg bg-violet-950 text-violet-300">
          <FileText className="document-card-icon h-5 w-5" />
        </div>

        <div className="document-card-copy min-w-0 flex-1">
          <h2 className="document-card-title truncate text-sm font-semibold text-white">
            {document.original_filename || document.filename}
          </h2>
          <p className="document-card-meta mt-1 text-[10px] text-slate-500">
            {formatBytes(document.file_size)} • {formatDate(document.created_at)}
          </p>
        </div>

        <StatusPill status={document.status || "ready"} />

        <Button
          variant="secondary"
          loading={downloading}
          onClick={() => void download()}
        >
          <Download className="h-4 w-4" />
          Download
        </Button>
      </Card>

      <Card className="min-h-[500px] overflow-hidden">
        {isPdf && previewUrl ? (
          <iframe
            title="Document viewer"
            src={previewUrl}
            className="h-[72vh] min-h-[500px] w-full bg-white"
          />
        ) : (
          <div className="grid min-h-[500px] place-items-center p-8 text-center">
            <div>
              <FileText
                data-document-type={documentType}
                className="document-card-icon mx-auto h-12 w-12 text-slate-700"
              />
              <h3 className="mt-4 text-sm font-semibold text-slate-300">
                Preview via download
              </h3>
              <p className="mt-1 max-w-md text-xs text-slate-500">
                Browser-native preview is provided for PDFs. DOCX, TXT, CSV and
                XLSX are available through the authenticated backend download
                endpoint and the RAG interfaces.
              </p>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
