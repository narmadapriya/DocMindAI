from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.rag.vectordb.retriever import Retriever


class RetrievalAgent:
    """
    LangGraph-compatible Retrieval Agent.

    Converts the query plan into a retrieval request and
    returns normalized evidence.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
    ):
        self.retriever = (
            retriever or Retriever()
        )

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        document_ids: Optional[
            Sequence[str]
        ] = None,
        modalities: Optional[
            Sequence[str]
        ] = None,
        metadata_filters: Optional[
            Dict[str, Any]
        ] = None,
        top_k: int = 5,
        min_relevance: float = 0.20,
    ) -> List[Dict[str, Any]]:

        results = self.retriever.retrieve(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            modalities=modalities,
            metadata_filters=metadata_filters,
            top_k=top_k,
            min_relevance=min_relevance,
        )

        # Always return a list.
        if not results:
            return []

        return [
            result
            for result in results
            if isinstance(
                result,
                dict,
            )
        ]