from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from app.main_services import (
    get_reasoning_agent,
)

from app.models.chat import (
    Chat,
    Message,
)

from app.models.document import (
    Document,
)

from app.models.user import (
    User,
)

from app.rag.agents.comparison_agent import (
    ComparisonAgent,
)

from app.rag.agents.summary_agent import (
    SummaryAgent,
)

from app.services.agentic_rag_service import (
    AgenticRAGService,
    VerifiedRetrievalGraphAdapter,
)

from app.services.analytics_service import (
    AnalyticsService,
)

from app.services.comparison_service import (
    ComparisonService,
)

from app.services.retrieval_service import (
    RetrievalService,
)

from app.services.summary_service import (
    SummaryService,
)


class RAGApplicationService:
    """
    Step 9 application layer for:

        POST /api/v1/chat
        POST /api/v1/summary
        POST /api/v1/compare

    Production flow
    ---------------

    CHAT:
        API
          -> PostgreSQL Chat
          -> AgenticRAGService
          -> Step 7 RetrievalService
          -> Phase 10 reasoning/verification/citations
          -> PostgreSQL Messages/Citations
          -> Analytics

    SUMMARY:
        API
          -> Step 7 RetrievalService
          -> existing Phase 11 SummaryService
          -> Analytics

    COMPARISON:
        API
          -> Step 7 RetrievalService
          -> existing Phase 11 ComparisonService
          -> Analytics
    """

    def __init__(
        self,
        db: Session,
        *,
        retrieval_service:
            RetrievalService
            | None = None,
        agentic_rag_service:
            AgenticRAGService
            | None = None,
        summary_service:
            SummaryService
            | None = None,
        comparison_service:
            ComparisonService
            | None = None,
        analytics_service:
            AnalyticsService
            | None = None,
    ):
        self.db = db

        # -----------------------------------------------------
        # Step 7 secure retrieval boundary
        # -----------------------------------------------------

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

        # Reuse the process-wide frozen ReasoningAgent/Ollama client.
        # The retrieval adapter remains request/DB scoped, so ownership
        # and PostgreSQL verification boundaries are unchanged.
        shared_reasoning_agent = (
            get_reasoning_agent()
        )

        # -----------------------------------------------------
        # Step 8 Chat / Agentic RAG
        # -----------------------------------------------------

        self.agentic_rag = (
            agentic_rag_service
            or AgenticRAGService(
                db,
                retrieval_service=(
                    self.retrieval_service
                ),
                reasoning_agent=(
                    shared_reasoning_agent
                ),
            )
        )

        # -----------------------------------------------------
        # Frozen Phase 11 Summary/Comparison agents,
        # now backed by Step 7 verified retrieval.
        # -----------------------------------------------------

        if (
            summary_service
            is None
            or comparison_service
            is None
        ):
            reasoning_agent = (
                shared_reasoning_agent
            )

        self.summary_service = (
            summary_service
            or SummaryService(
                summary_agent=(
                    SummaryAgent(
                        reasoning_agent=(
                            reasoning_agent
                        )
                    )
                ),
                retrieval_graph=(
                    self.retrieval_adapter
                ),
            )
        )

        self.comparison_service = (
            comparison_service
            or ComparisonService(
                comparison_agent=(
                    ComparisonAgent(
                        reasoning_agent=(
                            reasoning_agent
                        )
                    )
                ),
                retrieval_graph=(
                    self.retrieval_adapter
                ),
            )
        )

        # -----------------------------------------------------
        # PostgreSQL analytics
        # -----------------------------------------------------

        self.analytics = (
            analytics_service
            or AnalyticsService(
                db
            )
        )

    # =========================================================
    # CHAT
    # =========================================================

    def chat(
        self,
        *,
        user: User,
        query: str,
        document_ids: Sequence[str],
        conversation_id: str | None = None,
        history: Sequence[
            dict[str, str]
        ] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:

        started = perf_counter()

        query = str(
            query
            or ""
        ).strip()

        if not query:
            raise ValueError(
                "Chat query cannot be empty."
            )

        document_ids = [
            str(value)
            for value
            in document_ids
        ]

        # -----------------------------------------------------
        # Validate requested PostgreSQL documents before
        # creating/persisting a new conversation.
        # -----------------------------------------------------

        self._validate_documents(
            user_id=user.id,
            document_ids=document_ids,
        )

        chat, created = (
            self._get_or_create_chat(
                user=user,
                query=query,
                conversation_id=(
                    conversation_id
                ),
            )
        )

        # -----------------------------------------------------
        # Legacy request history:
        #
        # For a newly-created conversation we can import
        # supplied history once into PostgreSQL.
        #
        # Existing conversations use PostgreSQL as the source
        # of truth and do not duplicate client-sent history.
        # -----------------------------------------------------

        if created and history:

            self._persist_initial_history(
                chat=chat,
                history=history,
            )

        persisted_history = (
            self._conversation_history(
                chat=chat,
            )
        )

        conversation_context_used = bool(
            persisted_history
        )

        # -----------------------------------------------------
        # Preserve Phase 11 follow-up behavior using persisted
        # conversation context.
        # -----------------------------------------------------

        contextual_query = (
            self._build_contextual_query(
                query=query,
                history=(
                    persisted_history
                ),
            )
        )

        # -----------------------------------------------------
        # Step 8:
        # retrieval -> validation -> reasoning ->
        # verification -> citation -> DB persistence
        # -----------------------------------------------------

        result = (
            self.agentic_rag.answer(
                user_id=str(
                    user.id
                ),
                chat_id=str(
                    chat.id
                ),
                query=(
                    contextual_query
                ),
                document_ids=(
                    document_ids
                ),
                top_k=top_k,
            )
        )

        # -----------------------------------------------------
        # AgenticRAGService persisted contextual_query as the
        # user message. Restore the actual user-visible query.
        # -----------------------------------------------------

        if (
            contextual_query
            != query
        ):

            user_message_id = (
                result.get(
                    "user_message_id"
                )
            )

            if user_message_id:

                user_message = (
                    self.db.query(
                        Message
                    )
                    .filter(
                        Message.id
                        == self._parse_uuid(
                            user_message_id,
                            field_name=(
                                "user_message_id"
                            ),
                        )
                    )
                    .first()
                )

                if user_message:

                    user_message.content = (
                        query
                    )

                    self.db.commit()

        # -----------------------------------------------------
        # Convert Phase 10 citation dictionaries into the
        # existing Phase 11 ChatResponse List[str].
        # -----------------------------------------------------

        citation_strings = (
            self._citation_strings(
                result.get(
                    "citations",
                    [],
                )
            )
        )

        evidence = list(
            result.get(
                "evidence",
                [],
            )
        )

        elapsed_ms = (
            self._elapsed_ms(
                started
            )
        )

        # -----------------------------------------------------
        # Analytics
        # -----------------------------------------------------

        self.analytics.record_event(
            user_id=user.id,
            event_type="rag_chat",
            document_ids=(
                document_ids
            ),
            query_count=1,
            response_time_ms=(
                elapsed_ms
            ),
        )

        return {
            "conversation_id": str(
                chat.id
            ),

            "answer": str(
                result.get(
                    "answer",
                    "",
                )
            ),

            "citations":
                citation_strings,

            "evidence_count":
                int(
                    result.get(
                        "evidence_count",
                        len(evidence),
                    )
                ),

            "document_count":
                self._document_count(
                    evidence
                ),

            "conversation_context_used":
                conversation_context_used,

            "assistant_message_id":
                result.get(
                    "assistant_message_id"
                ),

            "analytics_recorded":
                True,
        }

    # =========================================================
    # SUMMARY
    # =========================================================

    def summarize(
        self,
        *,
        user: User,
        document_ids: Sequence[str],
        scope: str = "document",
        section: str | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:

        started = perf_counter()

        document_ids = [
            str(value)
            for value
            in document_ids
        ]

        self._validate_documents(
            user_id=user.id,
            document_ids=document_ids,
        )

        result = (
            self.summary_service
            .summarize(
                user_id=str(
                    user.id
                ),
                document_ids=(
                    document_ids
                ),
                scope=scope,
                section=section,
                top_k=top_k,
            )
        )

        self.analytics.record_event(
            user_id=user.id,
            event_type=(
                "rag_summary"
            ),
            document_ids=(
                document_ids
            ),
            query_count=1,
            response_time_ms=(
                self._elapsed_ms(
                    started
                )
            ),
        )

        result[
            "analytics_recorded"
        ] = True

        return result

    # =========================================================
    # COMPARISON
    # =========================================================

    def compare(
        self,
        *,
        user: User,
        document_ids: Sequence[str],
        metrics: Sequence[str] | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:

        started = perf_counter()

        document_ids = [
            str(value)
            for value
            in document_ids
        ]

        if len(document_ids) < 2:

            raise ValueError(
                "Comparison requires at "
                "least two documents."
            )

        self._validate_documents(
            user_id=user.id,
            document_ids=document_ids,
        )

        result = (
            self.comparison_service
            .compare(
                user_id=str(
                    user.id
                ),
                document_ids=(
                    document_ids
                ),
                metrics=(
                    list(metrics)
                    if metrics
                    is not None
                    else None
                ),
                top_k=top_k,
            )
        )

        self.analytics.record_event(
            user_id=user.id,
            event_type=(
                "rag_comparison"
            ),
            document_ids=(
                document_ids
            ),
            query_count=1,
            response_time_ms=(
                self._elapsed_ms(
                    started
                )
            ),
        )

        result[
            "analytics_recorded"
        ] = True

        return result

    # =========================================================
    # DOCUMENT SECURITY
    # =========================================================

    def _validate_documents(
        self,
        *,
        user_id: UUID,
        document_ids: Sequence[str],
    ) -> None:

        if not document_ids:
            return

        parsed_ids = [
            self._parse_uuid(
                value,
                field_name="document_id",
            )
            for value
            in document_ids
        ]

        rows = (
            self.db.query(
                Document
            )
            .filter(
                Document.id.in_(
                    parsed_ids
                ),
                Document.owner_id
                == user_id,
            )
            .all()
        )

        authorized = {
            row.id
            for row in rows
        }

        requested = set(
            parsed_ids
        )

        if (
            authorized
            != requested
        ):

            raise PermissionError(
                "One or more selected "
                "documents do not exist "
                "or do not belong to "
                "the current user."
            )

    # =========================================================
    # CHAT
    # =========================================================

    def _get_or_create_chat(
        self,
        *,
        user: User,
        query: str,
        conversation_id: str | None,
    ) -> tuple[Chat, bool]:

        if conversation_id:

            chat_uuid = (
                self._parse_uuid(
                    conversation_id,
                    field_name=(
                        "conversation_id"
                    ),
                )
            )

            chat = (
                self.db.query(Chat)
                .filter(
                    Chat.id
                    == chat_uuid,
                    Chat.user_id
                    == user.id,
                )
                .first()
            )

            if chat is None:

                raise PermissionError(
                    "Conversation does not "
                    "exist or does not belong "
                    "to the current user."
                )

            return (
                chat,
                False,
            )

        title = (
            query[:80]
            if query
            else "New conversation"
        )

        chat = Chat(
            user_id=user.id,
            title=title,
        )

        self.db.add(
            chat
        )

        self.db.commit()

        self.db.refresh(
            chat
        )

        return (
            chat,
            True,
        )

    def _persist_initial_history(
        self,
        *,
        chat: Chat,
        history: Sequence[
            dict[str, str]
        ],
    ) -> None:

        allowed_roles = {
            "user",
            "assistant",
        }

        messages = []

        for item in history:

            role = str(
                item.get(
                    "role",
                    "",
                )
            ).strip().lower()

            content = str(
                item.get(
                    "content",
                    "",
                )
            ).strip()

            if (
                role
                not in allowed_roles
                or not content
            ):
                continue

            messages.append(
                Message(
                    chat_id=chat.id,
                    role=role,
                    content=content,
                )
            )

        if messages:

            self.db.add_all(
                messages
            )

            self.db.commit()

    @staticmethod
    def _conversation_history(
        *,
        chat: Chat,
    ) -> list[
        dict[str, str]
    ]:

        return [
            {
                "role":
                    message.role,

                "content":
                    message.content,
            }
            for message
            in list(
                chat.messages
            )[-6:]
        ]

    @staticmethod
    def _build_contextual_query(
        *,
        query: str,
        history: Sequence[
            dict[str, str]
        ],
    ) -> str:

        if not history:
            return query

        lines = [
            "Conversation history:"
        ]

        for item in history[-6:]:

            role = str(
                item.get(
                    "role",
                    "user",
                )
            ).upper()

            content = str(
                item.get(
                    "content",
                    "",
                )
            ).strip()

            if content:

                lines.append(
                    f"{role}: {content}"
                )

        lines.extend(
            [
                "",
                "Current user question:",
                query,
                "",
                (
                    "Resolve references such as "
                    "'it', 'that document', "
                    "'the previous number', or "
                    "'this value' from the "
                    "conversation history."
                ),
            ]
        )

        return "\n".join(
            lines
        )

    # =========================================================
    # CITATIONS
    # =========================================================

    @staticmethod
    def _citation_strings(
        citations: Sequence[Any],
    ) -> list[str]:

        values: list[str] = []

        seen: set[str] = set()

        for citation in citations:

            if isinstance(
                citation,
                dict,
            ):

                value = (
                    citation.get(
                        "text"
                    )
                    or citation.get(
                        "source"
                    )
                )

                if (
                    value
                    and not str(
                        value
                    ).startswith(
                        "[Source:"
                    )
                ):

                    value = (
                        f"[Source: "
                        f"{value}]"
                    )

            else:

                value = str(
                    citation
                )

            if not value:
                continue

            value = str(
                value
            )

            if value not in seen:

                seen.add(
                    value
                )

                values.append(
                    value
                )

        return values

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _document_count(
        evidence: Sequence[
            dict[str, Any]
        ],
    ) -> int:

        ids = {
            str(
                (
                    item.get(
                        "metadata",
                        {}
                    )
                    or {}
                ).get(
                    "document_id"
                )
            )
            for item
            in evidence
            if (
                item.get(
                    "metadata",
                    {}
                )
                or {}
            ).get(
                "document_id"
            )
        }

        return len(ids)

    @staticmethod
    def _elapsed_ms(
        started: float,
    ) -> int:

        return max(
            0,
            int(
                (
                    perf_counter()
                    - started
                )
                * 1000
            ),
        )

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
                f"{field_name} must "
                "be a valid UUID."
            ) from exc