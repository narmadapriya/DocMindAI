from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.rag.agents.graph import (
    RetrievalGraph,
)


class RetrievalPipeline:
    """
    Public Phase 9 retrieval pipeline.

    The application should call this class instead of
    directly interacting with LangGraph or ChromaDB.
    """

    def __init__(
        self,
        retrieval_graph: Optional[
            RetrievalGraph
        ] = None,
    ):

        self.graph = (
            retrieval_graph
            or RetrievalGraph()
        )

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        document_ids: Optional[
            List[str]
        ] = None,
        metadata_filters: Optional[
            Dict[str, Any]
        ] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        result = self.graph.invoke(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            metadata_filters=(
                metadata_filters
            ),
            top_k=top_k,
        )

        return {
            "query": result[
                "query"
            ],
            "rewritten_query": result.get(
                "rewritten_query",
                result["query"],
            ),
            "user_id": result[
                "user_id"
            ],
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
            "evidence": result.get(
                "evidence",
                [],
            ),
            "evidence_count": result.get(
                "evidence_count",
                0,
            ),
            "retrieval_status": result.get(
                "retrieval_status",
                "unknown",
            ),
        }