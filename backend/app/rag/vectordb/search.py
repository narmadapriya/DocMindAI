from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.rag.vectordb.chroma_manager import (
    ChromaManager,
)


class VectorSearch:
    """
    Phase 9 vector search layer.

    Responsibilities:
        - semantic vector search
        - metadata filtering
        - ChromaDB integration
        - distance normalization
        - relevance scoring
        - ranked retrieval
    """

    def __init__(
        self,
        chroma_manager: Optional[
            ChromaManager
        ] = None,
    ):
        self.chroma = (
            chroma_manager
            or ChromaManager()
        )

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metadata_filters: Optional[
            Dict[str, Any]
        ] = None,
    ) -> List[Dict[str, Any]]:

        if not query_embedding:
            return []

        top_k = max(1, int(top_k))

        result = self.chroma.query(
            query_embeddings=[
                list(query_embedding)
            ],
            n_results=top_k,
            where=(
                metadata_filters
                if metadata_filters
                else None
            ),
        )

        return self._normalize_results(result)

    # =========================================================
    # NORMALIZATION
    # =========================================================

    def _normalize_results(
        self,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        ids = self._first(
            result.get("ids")
        )

        documents = self._first(
            result.get("documents")
        )

        metadatas = self._first(
            result.get("metadatas")
        )

        distances = self._first(
            result.get("distances")
        )

        output: List[Dict[str, Any]] = []

        for index, chunk_id in enumerate(ids):

            content = (
                documents[index]
                if index < len(documents)
                else ""
            )

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            relevance = self.distance_to_relevance(
                distance
            )

            output.append(
                {
                    "chunk_id": chunk_id,
                    "content": content or "",
                    "metadata": metadata or {},
                    "distance": distance,
                    "relevance_score": relevance,
                }
            )

        output.sort(
            key=lambda item: item[
                "relevance_score"
            ],
            reverse=True,
        )

        return output

    # =========================================================
    # DISTANCE → RELEVANCE
    # =========================================================

    @staticmethod
    def distance_to_relevance(
        distance: float | None,
    ) -> float:

        if distance is None:
            return 0.0

        try:
            distance = float(distance)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        # Cosine distance:
        #
        # similarity = 1 - distance
        #
        # Clamp to [0, 1].

        relevance = 1.0 - distance

        return max(
            0.0,
            min(
                1.0,
                relevance,
            ),
        )

    # =========================================================
    # FIRST RESULT
    # =========================================================

    @staticmethod
    def _first(
        value: Any,
    ) -> List[Any]:

        if value is None:
            return []

        if isinstance(value, list):

            if not value:
                return []

            # Chroma query normally returns:
            #
            # [
            #     [item1, item2, ...]
            # ]
            #
            if isinstance(
                value[0],
                list,
            ):
                return value[0]

            return value

        return []