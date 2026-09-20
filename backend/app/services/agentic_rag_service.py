from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.performance import TTLCache

from app.models.chat import (
    Chat,
    Message,
)

from app.rag.agents.citation_agent import (
    CitationAgent,
)

from app.rag.agents.evidence_validation_agent import (
    EvidenceValidationAgent,
)

from app.rag.agents.graph import (
    MultimodalReasoningGraph,
)

from app.rag.agents.reasoning_agent import (
    ReasoningAgent,
)

from app.rag.agents.verification_agent import (
    VerificationAgent,
)

from app.services.citation_service import (
    CitationPersistenceService,
)

from app.services.retrieval_service import (
    RetrievalService,
)


# ============================================================
# VERIFIED ANSWER CACHE
# ============================================================

# Cache only final, verification-passed Agentic RAG results. The cache is
# process-local, bounded, and TTL-based through the existing Phase-13
# performance utility, so it requires no schema/API/authentication changes.
_VALID_CHAT_ANSWER_CACHE = TTLCache[dict[str, Any]]()


# ============================================================
# VERIFIED RETRIEVAL ADAPTER
# ============================================================

class VerifiedRetrievalGraphAdapter:
    """
    Adapter between:

        Step 7 RetrievalService

    and:

        existing Phase 10 MultimodalReasoningGraph.

    Phase 10 expects an object with:

        invoke(...)

    Step 7 exposes:

        retrieve(...)

    This adapter allows Phase 10 to consume PostgreSQL-verified
    evidence without modifying the frozen Phase 10 graph.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
    ):
        self.retrieval_service = (
            retrieval_service
        )

    @staticmethod
    def _is_explicit_detail_list_query(query: str) -> bool:
        """
        Return True only for explicit list/detail requests.

        This guard is intentionally narrow so ordinary Chat & Ask questions
        keep the frozen retrieval depth and behavior.
        """
        q = re.sub(r"\s+", " ", str(query or "").lower()).strip()
        if not q:
            return False

        asks_for_list = any(
            term in q
            for term in (
                "types",
                "type of",
                "kinds",
                "kind of",
                "categories",
                "category",
                "components",
                "features",
                "steps",
                "list",
            )
        )
        asks_for_detail = any(
            term in q
            for term in (
                "explain",
                "describe",
                "in detail",
                "detail",
                "what are",
                "give",
                "list",
            )
        )
        return asks_for_list and asks_for_detail

    @staticmethod
    def _retrieval_result_is_pdf(result: dict[str, Any]) -> bool:
        """
        Detect PDF evidence from the already verified retrieval result.

        No database lookup or API/schema change is introduced.
        """
        for item in list(result.get("evidence", []) or []):
            if not isinstance(item, dict):
                continue
            metadata = dict(item.get("metadata") or {})
            values = (
                item.get("file_type"),
                item.get("filename"),
                item.get("source"),
                metadata.get("file_type"),
                metadata.get("filename"),
                metadata.get("source"),
                metadata.get("source_file"),
                metadata.get("extension"),
            )
            for value in values:
                normalized = str(value or "").strip().lower()
                if normalized == "pdf" or normalized.endswith(".pdf"):
                    return True
        return False

    def invoke(
        self,
        *,
        query: str,
        user_id: str,
        document_ids: Optional[
            list[str]
        ] = None,
        metadata_filters: Optional[
            dict[str, Any]
        ] = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Execute Step 7 retrieval with PostgreSQL verification.
        """

        requested_top_k = max(1, int(top_k))

        result = (
            self.retrieval_service
            .retrieve(
                query=query,
                user_id=user_id,
                document_ids=(
                    document_ids
                    or None
                ),
                metadata_filters=(
                    metadata_filters
                ),
                top_k=requested_top_k,
                verify_postgres=True,
            )
        )

        # -------------------------------------------------
        # PDF DETAIL/LIST EVIDENCE-COMPLETENESS GUARD
        # -------------------------------------------------
        # Normal Chat & Ask retrieval remains exactly as before.  Only when
        # the user explicitly asks for a detailed/list-style answer AND the
        # first verified retrieval result is a PDF do we widen the existing
        # Step-7 top-k window.  This is required for source lists that span
        # several adjacent PDF chunks/pages (for example, Unit 1.pdf has the
        # seven network-device types split across page-11/page-12 chunks).
        #
        # Ranking, embeddings, Chroma filters, authentication, document
        # scoping, PostgreSQL verification, LangGraph topology, and every
        # non-PDF request remain unchanged.
        if (
            requested_top_k < 12
            and self._is_explicit_detail_list_query(query)
            and self._retrieval_result_is_pdf(result)
        ):
            expanded_result = (
                self.retrieval_service
                .retrieve(
                    query=query,
                    user_id=user_id,
                    document_ids=(
                        document_ids
                        or None
                    ),
                    metadata_filters=(
                        metadata_filters
                    ),
                    top_k=12,
                    verify_postgres=True,
                )
            )

            if len(expanded_result.get("evidence", []) or []) > len(
                result.get("evidence", []) or []
            ):
                result = expanded_result

        # Phase 10-compatible state.
        return {
            "query": result.get(
                "query",
                query,
            ),

            "rewritten_query":
                result.get(
                    "rewritten_query",
                    query,
                ),

            "user_id": user_id,

            "intent": result.get(
                "intent",
                "chat",
            ),

            "scope": result.get(
                "scope",
            ),

            "modalities": result.get(
                "modalities",
                [],
            ),

            "evidence": result.get(
                "evidence",
                [],
            ),

            "evidence_count":
                result.get(
                    "evidence_count",
                    0,
                ),

            "retrieval_status":
                result.get(
                    "retrieval_status",
                    "no_evidence",
                ),
        }


# ============================================================
# AGENTIC RAG SERVICE
# ============================================================

class AgenticRAGService:
    """
    Step 8 Agentic RAG orchestration service.

    Complete flow
    -------------

        Question
            ↓
        Step 7 RetrievalService
            ↓
        PostgreSQL verified evidence
            ↓
        EvidenceValidationAgent
            ↓
        ReasoningAgent
            ↓
        VerificationAgent
            ↓
        retry when required
            ↓
        CitationAgent
            ↓
        CitationWhitelist
            ↓
        PostgreSQL Message
            ↓
        PostgreSQL Citation rows

    This service does not rewrite the existing Phase 10 graph.
    It connects that graph to the Step 7 security boundary and
    PostgreSQL persistence.
    """

    def __init__(
        self,
        db: Session,
        *,
        retrieval_service:
            RetrievalService
            | None = None,
        reasoning_graph:
            MultimodalReasoningGraph
            | None = None,
        validation_agent:
            EvidenceValidationAgent
            | None = None,
        reasoning_agent:
            ReasoningAgent
            | None = None,
        verification_agent:
            VerificationAgent
            | None = None,
        citation_agent:
            CitationAgent
            | None = None,
        citation_service:
            CitationPersistenceService
            | None = None,
        max_retries: int = 1,
    ):
        self.db = db

        self.retrieval_service = (
            retrieval_service
            or RetrievalService(
                db
            )
        )

        self.retrieval_adapter = (
            VerifiedRetrievalGraphAdapter(
                self.retrieval_service
            )
        )

        self.reasoning_graph = (
            reasoning_graph
            or MultimodalReasoningGraph(
                retrieval_graph=(
                    self.retrieval_adapter
                ),
                validation_agent=(
                    validation_agent
                ),
                reasoning_agent=(
                    reasoning_agent
                ),
                verification_agent=(
                    verification_agent
                ),
                citation_agent=(
                    citation_agent
                ),
                max_retries=(
                    max_retries
                ),
            )
        )

        self.citation_service = (
            citation_service
            or CitationPersistenceService(
                db
            )
        )

    # =========================================================
    # ANSWER
    # =========================================================

    def answer(
        self,
        *,
        user_id: str,
        chat_id: str,
        query: str,
        document_ids: Optional[
            list[str]
        ] = None,
        metadata_filters: Optional[
            dict[str, Any]
        ] = None,
        top_k: int = 5,
        max_retries: Optional[
            int
        ] = None,
    ) -> dict[str, Any]:
        """
        Execute complete Step 8 Agentic RAG and persist the
        conversation + final citations.
        """

        if not query or not query.strip():

            raise ValueError(
                "query cannot be empty."
            )

        user_uuid = self._parse_uuid(
            user_id,
            field_name="user_id",
        )

        chat_uuid = self._parse_uuid(
            chat_id,
            field_name="chat_id",
        )

        # -----------------------------------------------------
        # Chat/user ownership boundary
        # -----------------------------------------------------

        chat = (
            self.db.query(Chat)
            .filter(
                Chat.id == chat_uuid,
                Chat.user_id
                == user_uuid,
            )
            .first()
        )

        if chat is None:

            raise PermissionError(
                "Chat does not exist or does "
                "not belong to the current user."
            )

        # -----------------------------------------------------
        # Agentic pipeline
        # -----------------------------------------------------

        graph_kwargs: dict[
            str,
            Any,
        ] = {
            "query": query.strip(),
            "user_id": str(
                user_uuid
            ),
            "document_ids": (
                document_ids
                or []
            ),
            "metadata_filters": (
                metadata_filters
                or {}
            ),
            "top_k": max(
                1,
                int(top_k),
            ),
        }

        if max_retries is not None:
            graph_kwargs[
                "max_retries"
            ] = max(
                0,
                int(
                    max_retries
                ),
            )

        cache_key = self._answer_cache_key(
            user_id=str(user_uuid),
            chat_id=str(chat_uuid),
            query=query,
            document_ids=(document_ids or []),
            metadata_filters=(metadata_filters or {}),
            top_k=max(1, int(top_k)),
        )

        cached_result = _VALID_CHAT_ANSWER_CACHE.get(cache_key)

        if cached_result is not None:
            # Reuse only a previously verification-passed result for the same
            # user/chat/question/document context. Message/citation persistence
            # below still runs normally, so conversation history and database
            # behavior remain unchanged while retrieval/LangGraph/Qwen are
            # skipped for the duplicate question.
            result = deepcopy(cached_result)
            result["answer_cache_hit"] = True
        else:
            result = (
                self.reasoning_graph
                .invoke(
                    **graph_kwargs
                )
            )

            verification = result.get("verification") or {}
            if (
                result.get("final_status") == "success"
                and verification.get("passed") is True
                and str(result.get("answer", "")).strip()
            ):
                cached_copy = deepcopy(result)
                cached_copy["answer_cache_hit"] = False
                _VALID_CHAT_ANSWER_CACHE.set(cache_key, cached_copy)

        generated_citations = list(
            result.get(
                "citations",
                [],
            )
        )

        evidence = list(
            result.get(
                "evidence",
                [],
            )
        )

        answer_text = str(
            result.get(
                "answer",
                "",
            )
        ).strip()

        if (
            result.get("final_status") == "success"
            and answer_text
        ):
            # Presentation is applied only after the existing LangGraph
            # verification/citation pipeline has completed successfully. This
            # keeps page numbers and markdown out of VerificationAgent numeric
            # grounding while still giving the frontend the requested direct,
            # readable, inline-cited response.
            answer_text = self._format_verified_chat_presentation(
                query=query,
                answer=answer_text,
                citations=generated_citations,
                evidence=evidence,
            )
            result["answer"] = answer_text

        if not answer_text:

            answer_text = (
                "I could not produce "
                "a verified answer."
            )

        # -----------------------------------------------------
        # Persist Message + Citations atomically.
        # -----------------------------------------------------

        try:

            user_message = Message(
                chat_id=chat.id,
                role="user",
                content=query.strip(),
            )

            assistant_message = Message(
                chat_id=chat.id,
                role="assistant",
                content=answer_text,
            )

            self.db.add(
                user_message
            )

            self.db.add(
                assistant_message
            )

            # Generate UUIDs first so Citation FK values can
            # reference the assistant message.
            self.db.flush()

            persisted_citations = (
                self.citation_service
                .persist_citations(
                    message=(
                        assistant_message
                    ),
                    generated_citations=(
                        generated_citations
                    ),
                    evidence=evidence,
                    replace_existing=True,
                    strict=True,
                    commit=False,
                )
            )

            self.db.commit()

            self.db.refresh(
                user_message
            )

            self.db.refresh(
                assistant_message
            )

            for citation in (
                persisted_citations
            ):
                self.db.refresh(
                    citation
                )

        except Exception:

            self.db.rollback()

            raise

        # -----------------------------------------------------
        # Final application result
        # -----------------------------------------------------

        response = dict(
            result
        )

        response.update(
            {
                "chat_id": str(
                    chat.id
                ),

                "user_message_id": str(
                    user_message.id
                ),

                "assistant_message_id":
                    str(
                        assistant_message.id
                    ),

                "persisted_citations": [
                    self.citation_service
                    .to_dict(
                        citation
                    )
                    for citation
                    in persisted_citations
                ],

                "persisted_citation_count":
                    len(
                        persisted_citations
                    ),

                "database_persisted":
                    True,
            }
        )

        return response

    # =========================================================
    # VERIFIED CHAT PRESENTATION (POST-VERIFICATION ONLY)
    # =========================================================

    @classmethod
    def _format_verified_chat_presentation(
        cls,
        *,
        query: str,
        answer: str,
        citations: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> str:
        """
        Convert an already-verified answer into the final Chat & Ask markdown
        contract. No retrieval, model call, verification rule, citation
        whitelist, API schema, or persistence behaviour is changed.

        Inline page/table/sheet identifiers are added *after* verification so
        citation numbers cannot be mistaken for answer numbers by the strict
        numeric verifier.
        """
        clean = cls._strip_chat_source_markers(answer)
        if not clean:
            return clean

        direct, support, items = cls._chat_answer_parts(clean)
        if not direct:
            direct = clean.strip()

        direct = cls._ensure_terminal_punctuation(direct)
        direct_citation = (
            cls._render_inline_citation(citations)
            if len(items) >= 2 and citations
            else cls._inline_citation_for_claim(
                direct,
                citations=citations,
                evidence=evidence,
            )
        )

        lines: list[str] = [
            "### Answer",
            "",
            f"**{direct}**{(' ' + direct_citation) if direct_citation else ''}",
        ]

        # Two or three short evidence sentences are ideal when they already
        # exist in the verified answer. Never invent filler merely to satisfy
        # the template.
        for sentence in support[:3]:
            sentence = cls._ensure_terminal_punctuation(sentence)
            citation = cls._inline_citation_for_claim(
                sentence,
                citations=citations,
                evidence=evidence,
            )
            lines.extend([
                "",
                f"{cls._emphasize_chat_metrics(sentence)}{(' ' + citation) if citation else ''}",
            ])

        if items:
            lines.extend(["", "#### Key Points"])
            for item in items:
                item = cls._compact_chat_list_item(item)
                item = cls._ensure_terminal_punctuation(item)
                citation = cls._inline_citation_for_claim(
                    item,
                    citations=citations,
                    evidence=evidence,
                )
                rendered = cls._emphasize_chat_metrics(item, bold_label=True)
                lines.append(
                    f"- {rendered}{(' ' + citation) if citation else ''}"
                )

        lines.extend([
            "",
            "**Next:** Ask for a comparison, a deeper explanation, or a source-specific breakdown if you want to explore this further.",
        ])
        return "\n".join(lines).strip()

    @staticmethod
    def _strip_chat_source_markers(value: str) -> str:
        text = str(value or "")
        text = re.sub(r"\s*\[Source:\s*[^\]]+\]", "", text, flags=re.I)
        text = re.sub(r"(?m)^\s*ANSWER\s*$", "", text, flags=re.I)
        text = re.sub(r"(?m)^\s*KEY\s+FINDINGS\s*$", "", text, flags=re.I)
        text = re.sub(r"(?m)^\s*[─━-]{8,}\s*$", "", text)
        return text.strip()

    @classmethod
    def _chat_answer_parts(
        cls,
        value: str,
    ) -> tuple[str, list[str], list[str]]:
        text = str(value or "").replace("\r", "\n").strip()
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]

        numbered: list[str] = []
        bullets: list[str] = []
        prose_lines: list[str] = []
        in_key_findings = False

        for line in raw_lines:
            if re.fullmatch(r"KEY\s+FINDINGS", line, flags=re.I):
                in_key_findings = True
                continue
            numbered_match = re.match(r"^\d{1,2}\s*[.)]\s*(.+)$", line)
            if numbered_match:
                numbered.append(numbered_match.group(1).strip())
                continue
            bullet_match = re.match(r"^[•*\-]\s+(.+)$", line)
            if bullet_match:
                bullets.append(bullet_match.group(1).strip())
                continue
            if in_key_findings:
                # A flattened markdown bullet can arrive as ordinary text after
                # transport. Keep it as a finding rather than main prose.
                bullets.append(line.strip(" •*-"))
            else:
                prose_lines.append(line)

        prose = " ".join(prose_lines)
        sentences = [
            re.sub(r"\s+", " ", part).strip()
            for part in re.split(r"(?<=[.!?])\s+", prose)
            if re.sub(r"\s+", " ", part).strip()
        ]
        direct = sentences[0] if sentences else (prose_lines[0] if prose_lines else "")
        support = sentences[1:4]

        # Prefer the complete explicit list produced by the grounded list
        # completion helper. KEY FINDINGS are only a fallback for answers that
        # genuinely have no complete numbered list.
        items = numbered if numbered else bullets

        # Remove list-introduction boilerplate and duplicate direct/support
        # claims without changing factual text.
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            item = re.sub(r"\s+", " ", str(item or "")).strip(" •*-\t")
            if not item:
                continue
            key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return direct.strip(), support, deduped

    @staticmethod
    def _compact_chat_list_item(value: str) -> str:
        """
        Keep an already-verified list item concise and scan-friendly.

        This is presentation-only and runs after verification.  It preserves
        the source wording, item label, and the first complete explanatory
        idea(s); it never invents or substitutes facts.
        """
        text = re.sub(r"\s+", " ", str(value or "")).strip(" •*-\t")
        if not text:
            return text

        label = ""
        body = text
        if ":" in text:
            possible_label, possible_body = text.split(":", 1)
            if 1 <= len(possible_label.split()) <= 7:
                label = possible_label.strip()
                body = possible_body.strip()

        # PDF textbook extraction often flattens several ideas into one very
        # long sentence.  Split only at punctuation / explicit transition
        # boundaries that already exist in the verified source wording.
        clauses = [
            re.sub(r"\s+", " ", part).strip(" ;")
            for part in re.split(
                r"(?<=[.!?])\s+|;\s*|\s+(?=(?:Advantages|Benefits|Drawbacks|Disadvantages|Conversely)\b)",
                body,
                flags=re.I,
            )
            if re.sub(r"\s+", " ", part).strip(" ;")
        ]

        if not clauses:
            compact_body = body
        else:
            chosen: list[str] = []
            total_words = 0
            for clause in clauses:
                words = clause.split()
                if chosen and total_words + len(words) > 34:
                    break
                chosen.append(clause)
                total_words += len(words)
                if len(chosen) >= 2 or total_words >= 24:
                    break
            compact_body = ". ".join(
                clause.rstrip(".!?") for clause in chosen
            ).strip()

        if label:
            return f"{label}: {compact_body}".strip()
        return compact_body

    @staticmethod
    def _ensure_terminal_punctuation(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text and text[-1:] not in ".!?":
            text += "."
        return text

    @staticmethod
    def _emphasize_chat_metrics(
        value: str,
        *,
        bold_label: bool = False,
    ) -> str:
        text = str(value or "")
        if bold_label and ":" in text:
            label, rest = text.split(":", 1)
            if 1 <= len(label.split()) <= 7:
                text = f"**{label.strip()}:**{rest}"

        # Highlight explicit metrics/dates without touching numbers embedded in
        # citation tags (citations are appended only after this helper runs).
        metric_pattern = re.compile(
            r"(?<![\w*])(?:\$\s*)?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|percent|million|billion|thousand|units?|employees?|years?))?",
            re.I,
        )
        return metric_pattern.sub(lambda m: f"**{m.group(0)}**", text)

    @classmethod
    def _inline_citation_for_claim(
        cls,
        claim: str,
        *,
        citations: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> str:
        if not citations:
            return ""

        evidence_by_id = {
            str(item.get("chunk_id")): item
            for item in evidence
            if isinstance(item, dict) and item.get("chunk_id") is not None
        }
        claim_tokens = cls._citation_tokens(claim)
        claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", str(claim or "").replace(",", "")))

        ranked: list[tuple[float, dict[str, Any]]] = []
        for citation in citations:
            evidence_id = str(citation.get("evidence_id") or "")
            item = evidence_by_id.get(evidence_id, {})
            content = str(item.get("content", item.get("text", "")) or "")
            content_tokens = cls._citation_tokens(content)
            overlap = len(claim_tokens & content_tokens) / max(1, len(claim_tokens))
            content_numbers = set(re.findall(r"\d+(?:\.\d+)?", content.replace(",", "")))
            number_score = (
                len(claim_numbers & content_numbers) / max(1, len(claim_numbers))
                if claim_numbers else 0.0
            )
            ranked.append((overlap + (0.8 * number_score), citation))

        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            return ""

        best_score = ranked[0][0]
        selected = [ranked[0][1]]
        if len(ranked) > 1 and best_score > 0:
            second_score = ranked[1][0]
            if second_score >= max(0.20, best_score * 0.60):
                selected.append(ranked[1][1])

        return cls._render_inline_citation(selected)

    @staticmethod
    def _citation_tokens(value: str) -> set[str]:
        stop = {
            "the", "and", "for", "with", "from", "that", "this", "are", "was",
            "were", "has", "have", "into", "only", "than", "more", "less", "its",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", str(value or "").lower())
            if token not in stop
        }

    @classmethod
    def _render_inline_citation(
        cls,
        citations: list[dict[str, Any]],
    ) -> str:
        parsed: list[tuple[str, str, str]] = []
        for citation in citations:
            source = str(citation.get("source") or "").strip()
            if not source:
                continue
            match = re.match(r"^(.*?),\s*Page\s+(\d+)\s*$", source, flags=re.I)
            if match:
                parsed.append((match.group(1).strip(), "page", match.group(2)))
                continue
            match = re.match(r"^(.*?),\s*Table\s+(.+)$", source, flags=re.I)
            if match:
                parsed.append((match.group(1).strip(), "table", match.group(2).strip()))
                continue
            match = re.match(r"^(.*?),\s*Sheet:\s*(.+)$", source, flags=re.I)
            if match:
                parsed.append((match.group(1).strip(), "sheet", match.group(2).strip()))
                continue
            parsed.append((source, "document", ""))

        if not parsed:
            return ""

        filenames = {value[0] for value in parsed}
        if len(filenames) == 1:
            filename = parsed[0][0]
            pages = sorted({int(value[2]) for value in parsed if value[1] == "page"})
            if pages:
                if len(pages) == 1:
                    return f"[Page {pages[0]}]"
                if pages == list(range(pages[0], pages[-1] + 1)):
                    return f"[Pages {pages[0]}–{pages[-1]}]"
                return "[Pages " + ", ".join(str(value) for value in pages) + "]"
            for kind, label in (("table", "Table"), ("sheet", "Sheet")):
                values = [value[2] for value in parsed if value[1] == kind]
                if values:
                    return f"[{label} {values[0]}]"
            return f"[{filename}]"

        rendered: list[str] = []
        for filename, kind, value in parsed[:2]:
            if kind == "page":
                rendered.append(f"{filename} — Page {value}")
            elif kind == "table":
                rendered.append(f"{filename} — Table {value}")
            elif kind == "sheet":
                rendered.append(f"{filename} — Sheet {value}")
            else:
                rendered.append(filename)
        return "[" + "; ".join(rendered) + "]"

    # =========================================================
    # VERIFIED ANSWER CACHE
    # =========================================================

    @classmethod
    def _answer_cache_key(
        cls,
        *,
        user_id: str,
        chat_id: str,
        query: str,
        document_ids: list[str],
        metadata_filters: dict[str, Any],
        top_k: int,
    ) -> str:
        """
        Build an exact context key for a previously verified chat answer.

        RAGApplicationService can wrap a follow-up in persisted conversation
        history. For cache identity we use only the current visible question,
        plus user/chat/document context. This allows an exact repeat to return
        immediately while a different question or document selection always
        executes normal retrieval.
        """

        current_question = cls._current_question_for_cache(query)
        material = json.dumps(
            {
                "user_id": str(user_id),
                "chat_id": str(chat_id),
                "question": cls._normalize_cache_text(current_question),
                "document_ids": sorted(str(value) for value in document_ids),
                "metadata_filters": metadata_filters or {},
                "top_k": int(top_k),
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_cache_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().casefold()

    @staticmethod
    def _current_question_for_cache(query: str) -> str:
        text = str(query or "").strip()
        marker = "Current user question:"
        if marker not in text:
            return text

        tail = text.rsplit(marker, 1)[-1].strip()
        instruction = (
            "Resolve references such as "
        )
        if instruction in tail:
            tail = tail.split(instruction, 1)[0].strip()
        return tail or text

    # =========================================================
    # UUID
    # =========================================================

    @staticmethod
    def _parse_uuid(
        value: UUID | str,
        *,
        field_name: str,
    ) -> UUID:

        if isinstance(
            value,
            UUID,
        ):
            return value

        try:

            return UUID(
                str(value)
            )

        except (
            TypeError,
            ValueError,
            AttributeError,
        ) as exc:

            raise ValueError(
                f"{field_name} must be "
                "a valid UUID."
            ) from exc