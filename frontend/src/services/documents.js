import { api, API_BASE_URL } from "./api";

function normalizeDocument(data, defaultStatus = "uploaded") {
  const raw = data?.document || data?.data || data || {};
  const id = raw.id || raw.document_id || raw.uuid;

  return {
    ...raw,
    id,
    filename: raw.filename || raw.original_filename || raw.name || "document",
    original_filename: raw.original_filename || raw.filename || raw.name || "document",
    status: raw.status || (raw.ready_for_rag === true ? "ready" : defaultStatus),
  };
}

function unwrapDocuments(data) {
  const items = Array.isArray(data)
    ? data
    : data?.documents || data?.items || data?.data || [];

  return items
    .map((item) => normalizeDocument(item, "uploaded"))
    .filter((item) => item.id);
}

export const documentService = {
  async list() {
    const { data } = await api.get("/api/upload/documents");
    const documents = unwrapDocuments(data);

    // The list endpoint intentionally returns frozen document-model fields only.
    // Enrich the repository from the existing authenticated status endpoint so
    // status, chunks, vectors, and page metadata stay real and current.
    const statuses = await Promise.allSettled(
      documents.map((document) =>
        api.get(`/api/upload/status/${document.id}`),
      ),
    );

    return documents.map((document, index) => {
      const result = statuses[index];
      if (result?.status !== "fulfilled") {
        return document;
      }

      return {
        ...document,
        ...result.value.data,
        id: document.id,
        filename: document.filename,
        original_filename: document.original_filename,
        file_type: document.file_type,
        file_size: document.file_size,
        created_at: document.created_at,
      };
    });
  },

  async get(documentId) {
    const documents = await this.list();
    const found = documents.find(
      (item) => String(item.id) === String(documentId),
    );

    if (!found) {
      throw new Error("Document metadata could not be found.");
    }

    return found;
  },

  async upload(file, onProgress) {
    const form = new FormData();
    form.append("file", file, file.name);

    const { data } = await api.post("/api/upload/", form, {
      timeout: 600_000,
      onUploadProgress: (event) => {
        const total = event.total || file.size || 1;
        const percentage = Math.min(
          99,
          Math.round((event.loaded / total) * 100),
        );
        onProgress?.(percentage);
      },
    });

    return normalizeDocument(data, "processing");
  },

  async uploadBatch(files, onFileProgress) {
    const results = [];

    for (const file of files) {
      const created = await this.upload(
        file,
        (progress) => onFileProgress?.(file, progress),
      );
      results.push(created);
    }

    return results;
  },


  async status(documentId) {
    const { data } = await api.get(`/api/upload/status/${documentId}`);
    return data;
  },

  async reindex(documentId) {
    const { data } = await api.post(`/api/upload/reindex/${documentId}`);
    return data;
  },

  async remove(documentId) {
    await api.delete(`/api/upload/${documentId}`);
  },

  downloadUrl(documentId) {
    return `${API_BASE_URL}/api/upload/download/${documentId}`;
  },
};
