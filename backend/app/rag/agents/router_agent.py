from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryPlan:
    """
    Structured retrieval plan produced by Router Agent.
    """

    original_query: str

    rewritten_query: str

    intent: str = "chat"

    scope: str = "all_documents"

    document_ids: List[str] = field(
        default_factory=list
    )

    modalities: List[str] = field(
        default_factory=lambda: [
            "text",
            "table",
            "image",
            "chart",
        ]
    )

    metadata_filters: Dict[str, Any] = field(
        default_factory=dict
    )

    top_k: int = 5

    min_relevance: float = 0.20


class RouterAgent:
    """
    Phase 9 Router Agent.

    Responsibilities:

        - query planning
        - document scope
        - query rewriting
        - modality detection
        - metadata filter planning
    """

    def __init__(
        self,
        default_top_k: int = 5,
        min_relevance: float = 0.20,
    ):
        self.default_top_k = default_top_k
        self.min_relevance = min_relevance

    # =========================================================
    # PUBLIC
    # =========================================================

    def plan(
        self,
        query: str,
        *,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        metadata_filters: Optional[
            Dict[str, Any]
        ] = None,
    ) -> QueryPlan:

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        normalized = query.strip()

        intent = self._detect_intent(
            normalized
        )

        scope = (
            "specific_documents"
            if document_ids
            else self._detect_scope(
                normalized
            )
        )

        modalities = self._detect_modalities(
            normalized
        )

        rewritten = self.rewrite_query(
            normalized
        )

        filters = dict(
            metadata_filters or {}
        )

        # -----------------------------------------------------
        # SECURITY: mandatory user isolation
        # -----------------------------------------------------

        filters["user_id"] = user_id

        # -----------------------------------------------------
        # DOCUMENT SCOPE
        # -----------------------------------------------------

        if document_ids:
            filters["document_id"] = {
                "$in": list(document_ids)
            }

        return QueryPlan(
            original_query=normalized,
            rewritten_query=rewritten,
            intent=intent,
            scope=scope,
            document_ids=(
                list(document_ids)
                if document_ids
                else []
            ),
            modalities=modalities,
            metadata_filters=filters,
            top_k=(
                max(1, top_k)
                if top_k is not None
                else self.default_top_k
            ),
            min_relevance=self.min_relevance,
        )

    # =========================================================
    # QUERY REWRITE
    # =========================================================

    def rewrite_query(
        self,
        query: str,
    ) -> str:

        rewritten = query.strip()

        prefixes = [
            "please tell me ",
            "can you tell me ",
            "could you tell me ",
            "i want to know ",
            "what can you tell me about ",
        ]

        lower = rewritten.lower()

        for prefix in prefixes:

            if lower.startswith(prefix):

                rewritten = rewritten[
                    len(prefix):
                ]

                break

        rewritten = re.sub(
            r"\s+",
            " ",
            rewritten,
        ).strip()

        return rewritten

    # =========================================================
    # INTENT
    # =========================================================

    def _detect_intent(
        self,
        query: str,
    ) -> str:

        q = query.lower()

        if any(
            word in q
            for word in [
                "compare",
                "difference",
                "versus",
                "vs",
                "between",
            ]
        ):
            return "comparison"

        if any(
            word in q
            for word in [
                "summarize",
                "summary",
                "overview",
            ]
        ):
            return "summary"

        return "chat"

    # =========================================================
    # SCOPE
    # =========================================================

    def _detect_scope(
        self,
        query: str,
    ) -> str:

        q = query.lower()

        cross_document_terms = [
            "across documents",
            "between documents",
            "all documents",
            "multiple documents",
            "different documents",
            "compare documents",
            "both documents",
            "across the documents",
        ]

        if any(
            term in q
            for term in cross_document_terms
        ):
            return "multi_document"

        return "single_or_all"

    # =========================================================
    # MODALITY
    # =========================================================

    def _detect_modalities(
        self,
        query: str,
    ) -> List[str]:

        q = query.lower()

        modalities: List[str] = []

        # -----------------------------------------------------
        # Table
        # -----------------------------------------------------

        if any(
            word in q
            for word in [
                "table",
                "row",
                "column",
                "spreadsheet",
                "worksheet",
                "excel",
                "csv",
            ]
        ):
            modalities.append("table")

        # -----------------------------------------------------
        # Image
        # -----------------------------------------------------

        if any(
            word in q
            for word in [
                "image",
                "photo",
                "picture",
                "visual",
            ]
        ):
            modalities.append("image")

        # -----------------------------------------------------
        # Chart
        # -----------------------------------------------------

        if any(
            word in q
            for word in [
                "chart",
                "graph",
                "trend",
                "axis",
                "plot",
                "increase",
                "decrease",
            ]
        ):
            modalities.append("chart")

        # -----------------------------------------------------
        # Normal text query
        # -----------------------------------------------------

        if not modalities:
            modalities.append("text")

        # -----------------------------------------------------
        # Multimodal query
        # -----------------------------------------------------

        if any(
            modality in modalities
            for modality in [
                "table",
                "image",
                "chart",
            ]
        ):
            if "text" not in modalities:
                modalities.append("text")

        return list(
            dict.fromkeys(modalities)
        )