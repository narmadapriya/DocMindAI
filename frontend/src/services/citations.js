import { api } from "./api";
import { normalizeCitation } from "../utils/format";

function normalizeBackendCitation(raw, index = 0) {
  const normalized = normalizeCitation(raw, index) || {};
  const source = typeof raw === "object" && raw ? raw : {};

  return {
    ...normalized,
    ...source,
    id: source.id || source.citation_id || normalized.id,
    citation_id: source.citation_id || source.id || null,
    document_name:
      source.document_name ||
      source.filename ||
      normalized.filename ||
      null,
    filename:
      source.filename ||
      source.document_name ||
      normalized.filename ||
      null,
    page_number:
      source.page_number ?? source.page ?? normalized.page_number ?? null,
    confidence_score:
      source.confidence_score ?? source.relevance_score ?? null,
    relevance_score:
      source.relevance_score ?? source.confidence_score ?? null,
  };
}

export const citationService = {
  async list(limit = 100) {
    const { data } = await api.get(`/api/citations/?limit=${limit}`);
    const raw = Array.isArray(data)
      ? data
      : data?.citations || data?.items || [];

    return raw.map((item, index) =>
      normalizeBackendCitation(item, index),
    );
  },

  async context(citation) {
    const citationId = citation?.citation_id || citation?.id;
    const chunkId = citation?.chunk_id;

    if (citationId && !String(citationId).startsWith("citation-")) {
      const { data } = await api.get(`/api/citations/${citationId}`);
      return normalizeBackendCitation(data, 0);
    }

    if (chunkId) {
      const { data } = await api.get(`/api/chunks/${chunkId}`);
      return normalizeBackendCitation(data, 0);
    }

    throw new Error("This citation does not contain a persisted citation_id or chunk_id.");
  },
};
