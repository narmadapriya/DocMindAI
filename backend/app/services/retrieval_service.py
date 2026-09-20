from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document

from app.rag.pipelines.retrieval_pipeline import (
    RetrievalPipeline,
)
from app.core.performance import RAG_MAX_TOP_K, TTLCache
from app.core.logging import get_logger, log_event

logger = get_logger(__name__)
_RETRIEVAL_CACHE = TTLCache[dict]()


class RetrievalService:
    """
    Step 7 application-level retrieval integration.

    Responsibilities
    ----------------
    1. Receive a user question.
    2. Validate requested PostgreSQL documents belong to user.
    3. Call the existing Phase 9 RetrievalPipeline.
    4. Retrieve semantic evidence from ChromaDB.
    5. Validate returned Chroma evidence against PostgreSQL.
    6. Return normalized, verified evidence.

    Existing Phase 9 retrieval stack is preserved:

        RetrievalPipeline
            -> RetrievalGraph
            -> RouterAgent
            -> RetrievalAgent
            -> Retriever
            -> EmbeddingModel
            -> VectorSearch
            -> ChromaDB

    PostgreSQL remains the relational source of truth.
    ChromaDB remains the vector-search store.
    """

    def __init__(
        self,
        db: Session,
        *,
        retrieval_pipeline: RetrievalPipeline | None = None,
    ):
        self.db = db

        self.retrieval_pipeline = (
            retrieval_pipeline
            or RetrievalPipeline()
        )

    # =========================================================
    # PUBLIC RETRIEVAL
    # =========================================================

    def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        document_ids: Optional[
            Sequence[str]
        ] = None,
        metadata_filters: Optional[
            Dict[str, Any]
        ] = None,
        top_k: int = 5,
        verify_postgres: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve evidence for a user's question.

        Parameters
        ----------
        query:
            Natural-language user question.

        user_id:
            Current authenticated user's UUID.

        document_ids:
            Optional PostgreSQL document UUIDs.

            When supplied, ownership is verified before any
            vector search is performed.

        metadata_filters:
            Additional safe Chroma metadata filters.

        top_k:
            Maximum number of evidence chunks.

        verify_postgres:
            When True, returned Chroma evidence must correspond
            to a PostgreSQL chunk row.

            This should remain True in production.
        """

        started = perf_counter()

        # -----------------------------------------------------
        # Basic validation
        # -----------------------------------------------------

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if not user_id or not str(
            user_id
        ).strip():
            raise ValueError(
                "user_id cannot be empty."
            )

        top_k = min(RAG_MAX_TOP_K, max(1, int(top_k)))

        normalized_user_id = str(
            user_id
        )

        normalized_document_ids = [
            str(document_id)
            for document_id
            in (
                document_ids
                or []
            )
        ]

        cache_key = (normalized_user_id, tuple(sorted(normalized_document_ids)), query.strip(), top_k, repr(sorted((metadata_filters or {}).items())))
        if verify_postgres:
            cached = _RETRIEVAL_CACHE.get(cache_key)
            if cached is not None:
                log_event(
                    logger,
                    "retrieval_cache_hit",
                    user_id=normalized_user_id,
                    top_k=top_k,
                    latency_ms=round(
                        (perf_counter() - started) * 1000,
                        2,
                    ),
                )
                return dict(cached)

        # -----------------------------------------------------
        # Validate the current user UUID.
        # -----------------------------------------------------

        user_uuid = self._parse_uuid(
            normalized_user_id,
            field_name="user_id",
        )

        # -----------------------------------------------------
        # Document ownership verification
        # -----------------------------------------------------

        access_started = perf_counter()

        if normalized_document_ids:

            self._validate_document_access(
                user_id=user_uuid,
                document_ids=(
                    normalized_document_ids
                ),
            )

        access_ms = (
            perf_counter() - access_started
        ) * 1000

        # -----------------------------------------------------
        # Execute existing Phase 9 retrieval pipeline.
        # -----------------------------------------------------

        pipeline_started = perf_counter()

        result = (
            self.retrieval_pipeline
            .retrieve(
                query=query.strip(),
                user_id=normalized_user_id,
                document_ids=(
                    normalized_document_ids
                    or None
                ),
                metadata_filters=(
                    metadata_filters
                ),
                top_k=top_k,
            )
        )

        pipeline_ms = (
            perf_counter() - pipeline_started
        ) * 1000

        raw_evidence = list(
            result.get(
                "evidence",
                [],
            )
        )

        # -----------------------------------------------------
        # PostgreSQL verification
        # -----------------------------------------------------

        verify_started = perf_counter()

        if verify_postgres:

            evidence = (
                self._verify_evidence(
                    evidence=raw_evidence,
                    user_id=user_uuid,
                )
            )

        else:

            evidence = [
                self._normalize_evidence(
                    item
                )
                for item
                in raw_evidence
                if isinstance(
                    item,
                    dict,
                )
            ]

        verify_ms = (
            perf_counter() - verify_started
        ) * 1000

        # -----------------------------------------------------
        # Never exceed requested top_k after verification.
        # -----------------------------------------------------

        evidence = evidence[
            :top_k
        ]

        document_count = len(
            {
                str(
                    item.get(
                        "metadata",
                        {},
                    ).get(
                        "document_id",
                        "",
                    )
                )
                for item in evidence
                if item.get(
                    "metadata",
                    {},
                ).get(
                    "document_id"
                )
            }
        )

        # -----------------------------------------------------
        # Preserve Phase 9 pipeline information.
        # -----------------------------------------------------

        response = {
            "query": result.get(
                "query",
                query.strip(),
            ),

            "rewritten_query": (
                result.get(
                    "rewritten_query",
                    query.strip(),
                )
            ),

            "user_id": (
                normalized_user_id
            ),

            "intent": result.get(
                "intent"
            ),

            "scope": result.get(
                "scope"
            ),

            "modalities": result.get(
                "modalities",
                [],
            ),

            "document_ids": (
                normalized_document_ids
            ),

            "evidence": evidence,

            "evidence_count": len(
                evidence
            ),

            "document_count": (
                document_count
            ),

            "retrieval_status": (
                "success"
                if evidence
                else "no_evidence"
            ),

            "postgres_verified": (
                verify_postgres
            ),
        }
        if verify_postgres:
            _RETRIEVAL_CACHE.set(cache_key, dict(response))

        total_ms = (
            perf_counter() - started
        ) * 1000

        log_event(
            logger,
            "retrieval",
            user_id=normalized_user_id,
            evidence_count=len(evidence),
            document_count=document_count,
            top_k=top_k,
            access_ms=round(access_ms, 2),
            pipeline_ms=round(pipeline_ms, 2),
            postgres_verify_ms=round(verify_ms, 2),
            latency_ms=round(total_ms, 2),
        )
        return response

    # =========================================================
    # DOCUMENT ACCESS VALIDATION
    # =========================================================

    def _validate_document_access(
        self,
        *,
        user_id: UUID,
        document_ids: Sequence[str],
    ) -> None:
        """
        Confirm all requested documents exist and belong
        to the current user.
        """

        parsed_ids = [
            self._parse_uuid(
                document_id,
                field_name=(
                    "document_id"
                ),
            )
            for document_id
            in document_ids
        ]

        documents = (
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

        authorized_ids = {
            str(document.id)
            for document
            in documents
        }

        requested_ids = {
            str(document_id)
            for document_id
            in parsed_ids
        }

        unauthorized = (
            requested_ids
            - authorized_ids
        )

        if unauthorized:

            raise PermissionError(
                "One or more requested documents "
                "do not exist or do not belong "
                "to the current user."
            )

    # =========================================================
    # EVIDENCE VERIFICATION
    # =========================================================

    def _verify_evidence(
        self,
        *,
        evidence: Sequence[
            Dict[str, Any]
        ],
        user_id: UUID,
    ) -> List[Dict[str, Any]]:
        """
        Match Chroma evidence to PostgreSQL chunks.

        Primary relational key:

            document_id
                +
            chunk_index

        This works because Step 4 persisted the final
        ModalityAwareChunker indexes to PostgreSQL and Step 5
        stored the same chunk_index in Chroma metadata.
        """

        normalized: List[
            Dict[str, Any]
        ] = []

        # -----------------------------------------------------
        # First collect valid document/chunk-index pairs.
        # -----------------------------------------------------

        requested_pairs: List[
            tuple[UUID, int]
        ] = []

        for item in evidence:

            if not isinstance(
                item,
                dict,
            ):
                continue

            metadata = (
                item.get(
                    "metadata",
                    {}
                )
                or {}
            )

            document_id = (
                metadata.get(
                    "document_id"
                )
            )

            chunk_index = (
                metadata.get(
                    "chunk_index"
                )
            )

            if (
                document_id is None
                or chunk_index is None
            ):
                continue

            try:

                document_uuid = UUID(
                    str(
                        document_id
                    )
                )

                chunk_index_int = int(
                    chunk_index
                )

            except (
                ValueError,
                TypeError,
            ):
                continue

            requested_pairs.append(
                (
                    document_uuid,
                    chunk_index_int,
                )
            )

        if not requested_pairs:
            return []

        document_ids = list(
            {
                pair[0]
                for pair
                in requested_pairs
            }
        )

        # -----------------------------------------------------
        # Query only chunks belonging to documents owned
        # by the current user.
        # -----------------------------------------------------

        rows = (
            self.db.query(
                Chunk
            )
            .join(
                Document,
                Chunk.document_id
                == Document.id,
            )
            .filter(
                Chunk.document_id.in_(
                    document_ids
                ),
                Document.owner_id
                == user_id,
            )
            .all()
        )

        postgres_lookup = {
            (
                str(
                    row.document_id
                ),
                int(
                    row.chunk_index
                ),
            ):
            row

            for row in rows
        }

        # -----------------------------------------------------
        # Verify each Chroma result.
        # -----------------------------------------------------

        for item in evidence:

            if not isinstance(
                item,
                dict,
            ):
                continue

            normalized_item = (
                self._normalize_evidence(
                    item
                )
            )

            metadata = (
                normalized_item.get(
                    "metadata",
                    {}
                )
                or {}
            )

            document_id = (
                metadata.get(
                    "document_id"
                )
            )

            chunk_index = (
                metadata.get(
                    "chunk_index"
                )
            )

            if (
                document_id is None
                or chunk_index is None
            ):
                continue

            try:

                key = (
                    str(
                        UUID(
                            str(
                                document_id
                            )
                        )
                    ),
                    int(
                        chunk_index
                    ),
                )

            except (
                ValueError,
                TypeError,
            ):
                continue

            postgres_chunk = (
                postgres_lookup.get(
                    key
                )
            )

            # Stale or untrusted vector result.
            if postgres_chunk is None:
                continue

            # -------------------------------------------------
            # Verify essential metadata.
            # -------------------------------------------------

            chroma_chunk_type = str(
                metadata.get(
                    "chunk_type",
                    "",
                )
            ).lower()

            postgres_chunk_type = str(
                postgres_chunk.chunk_type
                or ""
            ).lower()

            if (
                chroma_chunk_type
                and postgres_chunk_type
                and chroma_chunk_type
                != postgres_chunk_type
            ):
                continue

            chroma_page = metadata.get(
                "page_number"
            )

            if (
                chroma_page is not None
                and postgres_chunk.page_number
                is not None
            ):

                try:

                    if int(
                        chroma_page
                    ) != int(
                        postgres_chunk.page_number
                    ):
                        continue

                except (
                    ValueError,
                    TypeError,
                ):
                    continue

            # -------------------------------------------------
            # Attach relational verification information.
            # -------------------------------------------------

            normalized_item[
                "postgres_chunk_id"
            ] = str(
                postgres_chunk.id
            )

            normalized_item[
                "postgres_verified"
            ] = True

            normalized_item[
                "content_matches_postgres"
            ] = (
                str(
                    normalized_item.get(
                        "content",
                        "",
                    )
                )
                == str(
                    postgres_chunk.content
                )
            )

            normalized.append(
                normalized_item
            )

        return normalized

    # =========================================================
    # EVIDENCE NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_evidence(
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return a predictable evidence structure for downstream
        Chat/Summary/Comparison agents.
        """

        metadata = dict(
            item.get(
                "metadata",
                {}
            )
            or {}
        )

        return {
            "chunk_id": item.get(
                "chunk_id"
            ),

            "content": str(
                item.get(
                    "content",
                    "",
                )
            ),

            "metadata": metadata,

            "distance": item.get(
                "distance"
            ),

            "relevance_score": float(
                item.get(
                    "relevance_score",
                    0.0,
                )
                or 0.0
            ),
        }

    # =========================================================
    # UUID
    # =========================================================

    @staticmethod
    def _parse_uuid(
        value: str,
        *,
        field_name: str,
    ) -> UUID:

        try:

            return UUID(
                str(value)
            )

        except (
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:

            raise ValueError(
                f"{field_name} must be "
                f"a valid UUID."
            ) from exc