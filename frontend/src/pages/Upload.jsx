import React from "react";
import { Card } from "../components/ui";
import { UploadDropzone } from "../components/UploadDropzone";
export default function Upload() {
    return (<div className="p-5">
      <Card className="mx-auto max-w-3xl p-6">
        <h2 className="text-base font-semibold text-white">Upload & Ingest Documents</h2>
        <p className="mt-1 mb-5 text-xs text-slate-500">Files are sent to the existing upload endpoint, then status is polled until the backend reports that the document is RAG-ready.</p>
        <UploadDropzone />
      </Card>
    </div>);
}
