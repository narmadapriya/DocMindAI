import { create } from "zustand";
import { ragService } from "../services/rag";
import { getErrorMessage } from "../services/api";
import { normalizeCitation } from "../utils/format";

function stripInlineSourceCitations(value) {
  return String(value || "")
    .replace(/\s*\[\s*Source\s*:\s*[^\]]+\]/gi, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

export const useChatStore = create((set, get) => ({
  messages: [],
  loading: false,
  error: null,
  chatId: undefined,

  clear: () =>
    set({
      messages: [],
      error: null,
      chatId: undefined,
    }),

  send: async ({
    userId,
    documentIds,
    query,
    topK,
  }) => {
    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
      createdAt: new Date().toISOString(),
    };

    set({
      messages: [
        ...get().messages,
        userMessage,
      ],
      loading: true,
      error: null,
    });

    try {
      const response = await ragService.chat({
        user_id: userId,
        document_ids: documentIds,
        query,
        top_k: topK,
        chat_id: get().chatId,
      });

      const citations = (
        response.citations || []
      ).map(normalizeCitation);

      const assistantMessage = {
        id:
          response.message_id ||
          crypto.randomUUID(),
        role: "assistant",

        // The backend returns answer and citations separately.
        // Strip any legacy/model-emitted [Source: ...] markers as
        // a UI safety net so sources render only in CitationList.
        content: stripInlineSourceCitations(
          response.answer,
        ),

        citations,
        createdAt: new Date().toISOString(),
      };

      set({
        messages: [
          ...get().messages,
          assistantMessage,
        ],
        loading: false,
        chatId:
          response.chat_id ||
          get().chatId,
      });
    } catch (error) {
      set({
        loading: false,
        error: getErrorMessage(error),
      });

      throw error;
    }
  },
}));