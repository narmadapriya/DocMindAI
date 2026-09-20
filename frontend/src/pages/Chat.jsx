import React from "react";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  Bot,
  Download,
  Pencil,
  Send,
  Sparkles,
  Trash2,
  UserRound,
} from "lucide-react";

import { Button } from "../components/ui";
import {
  CitationList,
  CitationViewer,
  RagSourcesPanel,
} from "../components/CitationPanel";
import { useDocumentStore } from "../store/documentStore";
import { useChatStore } from "../store/chatStore";
import { useAuthStore } from "../store/authStore";
import { documentReady } from "../utils/format";
import { downloadConversationPdf } from "../utils/pdfExport";

export default function Chat() {
  const user = useAuthStore((state) => state.user);
  const documents = useDocumentStore((state) => state.documents);
  const selectedIds = useDocumentStore((state) => state.selectedIds);
  const selectOne = useDocumentStore((state) => state.selectOne);
  const clearSelected = useDocumentStore((state) => state.clearSelected);
  const loadDocuments = useDocumentStore((state) => state.load);

  const messages = useChatStore((state) => state.messages);
  const loading = useChatStore((state) => state.loading);
  const error = useChatStore((state) => state.error);
  const send = useChatStore((state) => state.send);
  const clear = useChatStore((state) => state.clear);

  const [query, setQuery] = useState("");
  const [citation, setCitation] = useState(null);
  const [editingMessageId, setEditingMessageId] = useState(null);
  const bottom = useRef(null);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const readyDocs = useMemo(
    () => documents.filter(documentReady),
    [documents],
  );

  const active = readyDocs.find(
    (document) => String(document.id) === String(selectedIds[0]),
  );

  const clearAll = () => {
    clear();
    clearSelected();
    setQuery("");
    setEditingMessageId(null);
    setCitation(null);
  };

  const beginEdit = (message) => {
    setEditingMessageId(message.id);
    setQuery(message.content || "");
    requestAnimationFrame(() => {
      document.getElementById("chat-question-input")?.focus();
    });
  };

  const cancelEdit = () => {
    setEditingMessageId(null);
    setQuery("");
  };

  const submit = async (event) => {
    event.preventDefault();
    const question = query.trim();

    if (!question || !user || !active || loading) {
      return;
    }

    setQuery("");
    setEditingMessageId(null);

    try {
      await send({
        userId: user.id,
        documentIds: [active.id],
        query: question,
        topK: Number(import.meta.env.VITE_DEFAULT_TOP_K || 5),
      });
    } catch {
      setQuery(question);
    }
  };

  const exportConversation = () => {
    if (!messages.length) return;

    downloadConversationPdf({
      documentName: active?.original_filename || active?.filename || "",
      messages,
    });
  };

  return (
    <div className="chat-page flex min-h-[calc(100vh-60px)] bg-[#020718]">
      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex min-h-[52px] items-center justify-between gap-3 border-b border-slate-800 px-3 py-3 sm:px-4">
          <span className="conversation-agent-thread text-[14px] font-medium text-cyan-300">
            Conversation Agent Thread
          </span>

          <div className="flex items-center gap-1.5 sm:gap-2">
            <Button
              variant="secondary"
              className="chat-toolbar-action hidden px-3 py-2 text-xs sm:inline-flex"
              disabled={!messages.length}
              onClick={exportConversation}
              title="Download current conversation as PDF"
            >
              <Download className="h-3.5 w-3.5" />
              Download PDF
            </Button>

            <Button
              variant="ghost"
              className="chat-toolbar-action px-2.5 py-2 text-xs"
              onClick={clearAll}
              disabled={!messages.length && !active}
              title="Clear conversation and active document"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Clear Chat</span>
            </Button>
          </div>
        </div>

        <div className="border-b border-slate-800 p-3 lg:hidden">
          <label className="mb-1.5 block text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            RAG Source
          </label>
          <select
            className="field"
            value={active?.id || ""}
            onChange={(event) => selectOne(event.target.value || null)}
          >
            <option value="">Select a RAG-ready document</option>
            {readyDocs.map((document) => (
              <option key={document.id} value={document.id}>
                {document.original_filename || document.filename}
              </option>
            ))}
          </select>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-5 sm:px-5 lg:px-7 lg:py-7">
          <div className="mx-auto w-full max-w-[840px]">
            {!messages.length && (
              <div className="py-10 text-center sm:py-16">
                <div className="mx-auto grid h-10 w-10 place-items-center rounded-xl border border-violet-800 bg-violet-950/40 text-violet-300">
                  <Bot className="h-5 w-5" />
                </div>
                <h2 className="mt-4 text-base font-bold text-white sm:text-lg">
                  Analyze and Reason Over Your Documents
                </h2>
                <p className="mx-auto mt-2 max-w-xl text-[10px] leading-5 text-slate-500 sm:text-xs">
                  Select a document from the RAG Sources Panel, then ask a grounded question.
                </p>
              </div>
            )}

            {messages.map((message) => {
              const isUser = message.role === "user";
              const canEdit = isUser && !loading;

              return (
                <div
                  key={message.id}
                  className={`mb-5 ${
                    isUser
                      ? "ml-auto max-w-[96%] sm:max-w-[78%]"
                      : "mr-auto max-w-[98%] sm:max-w-[88%]"
                  }`}
                >
                  <div
                    className={`mb-1.5 flex items-center gap-1.5 text-[10px] font-medium ${
                      isUser ? "justify-end" : "justify-start"
                    }`}
                  >
                    <span
                      className={`grid h-6 w-6 shrink-0 place-items-center rounded-lg border shadow-sm ${
                        isUser
                          ? "border-blue-500/30 bg-blue-500/10 text-blue-400"
                          : "border-violet-500/30 bg-violet-500/10 text-violet-300"
                      }`}
                      aria-hidden="true"
                    >
                      {isUser ? (
                        <UserRound className="h-3.5 w-3.5" strokeWidth={2} />
                      ) : (
                        <Bot className="h-3.5 w-3.5" strokeWidth={2} />
                      )}
                    </span>

                    <span className="text-slate-500">
                      {isUser ? "You" : "DocMind AI"}
                    </span>
                  </div>

                  <div className="relative rounded-xl border border-slate-800 bg-ink-900 p-4 text-sm leading-6 text-slate-300">
                    {canEdit && (
                      <button
                        type="button"
                        onClick={() => beginEdit(message)}
                        className="absolute right-3 top-3 grid h-8 w-8 place-items-center rounded-lg border border-slate-700 bg-ink-800 text-slate-400 transition hover:border-violet-700 hover:text-violet-300"
                        title="Edit and resubmit this question"
                        aria-label="Edit question"
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </button>
                    )}

                    <div className={canEdit ? "pr-10" : ""}>
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content}
                      </ReactMarkdown>
                    </div>

                    {message.citations?.length > 0 && (
                      <CitationList
                        citations={message.citations}
                        onOpen={setCitation}
                      />
                    )}
                  </div>
                </div>
              );
            })}

            {loading && (
              <div className="mr-auto max-w-[88%] rounded-xl border border-slate-800 bg-ink-900 p-4 text-xs text-slate-400">
                <Sparkles className="mr-2 inline h-4 w-4 animate-pulse text-violet-400" />
                Retrieving evidence, reasoning, verifying, and generating citations…
              </div>
            )}

            <div ref={bottom} />
          </div>
        </div>

        <div className="border-t border-slate-800 bg-[#03091a] p-3 sm:p-4">
          <div className="mx-auto max-w-[840px]">
            <Button
              variant="secondary"
              className="chat-toolbar-action mb-3 w-full text-xs sm:hidden"
              disabled={!messages.length}
              onClick={exportConversation}
            >
              <Download className="h-3.5 w-3.5" />
              Download PDF
            </Button>

            {!active ? (
              <>
                <div className="mb-2 rounded-lg border border-amber-900/70 bg-amber-950/20 px-3 py-2 text-[10px] text-amber-300">
                  Please select a file from the RAG Sources Panel on the right to start querying.
                </div>

                <div className="relative">
                  <textarea
                    disabled
                    aria-disabled="true"
                    className="field min-h-[62px] cursor-not-allowed resize-none pr-14 opacity-40"
                    placeholder="Select a document from the RAG Sources Panel on the right..."
                    value=""
                    readOnly
                  />
                  <button
                    type="button"
                    disabled
                    className="absolute bottom-2.5 right-2.5 grid h-9 w-9 cursor-not-allowed place-items-center rounded-lg bg-slate-800 text-slate-600"
                    aria-label="Send disabled until a document is selected"
                  >
                    <Send className="h-4 w-4" />
                  </button>
                </div>
              </>
            ) : (
              <>
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-violet-700/60 bg-violet-950/25 px-3 py-2 text-[10px]">
                  <span className="min-w-0 truncate">
                    <span className="mr-2 inline-block h-2 w-2 rounded-full bg-emerald-400" />
                    RAG ACTIVE: <b>{active.original_filename || active.filename}</b>
                  </span>
                  <button
                    type="button"
                    className="rounded px-2 py-1 text-cyan-300 hover:bg-slate-800"
                    onClick={() => {
                      clearSelected();
                      setQuery("");
                      setEditingMessageId(null);
                    }}
                  >
                    Change File
                  </button>
                </div>

                {editingMessageId && (
                  <div className="mb-2 flex items-center justify-between rounded-lg border border-violet-800/70 bg-violet-950/20 px-3 py-2 text-[10px] text-violet-300">
                    <span>Editing question — submit to resubmit it as a new grounded turn.</span>
                    <button type="button" onClick={cancelEdit} className="font-semibold">
                      Cancel
                    </button>
                  </div>
                )}

                <form onSubmit={submit}>
                  <div className="relative">
                    <textarea
                      id="chat-question-input"
                      className="field min-h-[62px] resize-none pr-14"
                      placeholder="Ask anything about the selected document..."
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && !event.shiftKey) {
                          event.preventDefault();
                          event.currentTarget.form?.requestSubmit();
                        }
                      }}
                    />

                    <Button
                      type="submit"
                      disabled={!query.trim() || loading}
                      loading={loading}
                      className="absolute bottom-2.5 right-2.5 h-9 w-9 p-0"
                      aria-label="Send question"
                    >
                      <Send className="h-4 w-4" />
                    </Button>
                  </div>
                </form>

                {error && (
                  <div className="mt-2 rounded-lg border border-rose-900/70 bg-rose-950/20 p-2 text-[10px] text-rose-400">
                    {error}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </section>

      <RagSourcesPanel
        documents={readyDocs}
        selectedIds={selectedIds}
        onToggle={(id) =>
          selectOne(
            String(selectedIds[0]) === String(id)
              ? null
              : id,
          )
        }
      />

      <CitationViewer citation={citation} onClose={() => setCitation(null)} />
    </div>
  );
}
