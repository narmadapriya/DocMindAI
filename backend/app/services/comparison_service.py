from __future__ import annotations

from typing import Any, Dict, List, Sequence


class ComparisonService:
    """
    Phase 11 / Step 10 cross-document comparison service.

    The public API and application wiring are unchanged.

    Comparison-specific Phase-15 quality fix
    ----------------------------------------
    A comparison already knows exactly which documents the user selected.
    For numeric metrics, relying only on semantic top-k can omit a relevant
    table/row from one selected document even though that row is already
    indexed in ChromaDB. That produced false N/A values.

    This service therefore reads the existing indexed chunks for each selected
    document directly from the same Chroma collection when that collection is
    available through the normal RetrievalGraph. No document is reparsed, no
    new index is created, and Chat & Ask / Summary retrieval behavior is not
    changed.

    If direct indexed access is unavailable (for tests, alternate adapters, or
    a future retrieval implementation), the original per-document semantic
    retrieval path is used automatically.
    """

    def __init__(
        self,
        comparison_agent,
        retrieval_graph,
    ):
        self.comparison_agent = comparison_agent
        self.retrieval_graph = retrieval_graph

    # =========================================================
    # COMPARE
    # =========================================================

    def compare(
        self,
        *,
        user_id: str,
        document_ids: List[str],
        metrics: List[str] | None = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:

        normalized_document_ids = [
            str(document_id)
            for document_id in document_ids
            if str(document_id)
        ]
        normalized_document_ids = list(dict.fromkeys(normalized_document_ids))

        if len(normalized_document_ids) < 2:
            raise ValueError("Comparison requires at least two documents.")

        # -----------------------------------------------------
        # Fast path: use the already-indexed chunks belonging to
        # each selected document. This removes repeated embedding
        # calls and prevents false N/A caused by semantic top-k.
        # -----------------------------------------------------
        evidence = self._load_selected_indexed_evidence(
            user_id=str(user_id),
            document_ids=normalized_document_ids,
        )

        # -----------------------------------------------------
        # Safe compatibility fallback: retain the frozen
        # per-document semantic retrieval behavior whenever the
        # direct collection is not exposed by the supplied graph.
        # -----------------------------------------------------
        if not self._covers_all_documents(
            evidence=evidence,
            document_ids=normalized_document_ids,
        ):
            evidence = self._semantic_retrieval_fallback(
                user_id=str(user_id),
                document_ids=normalized_document_ids,
                metrics=metrics,
                top_k=top_k,
            )

        evidence = self._deduplicate_evidence(evidence)
        evidence = self._order_evidence(
            evidence=evidence,
            document_ids=normalized_document_ids,
        )

        result = self.comparison_agent.compare(
            evidence=evidence,
            document_ids=normalized_document_ids,
            metrics=metrics,
        )

        result["document_count"] = len(normalized_document_ids)
        return result

    # =========================================================
    # FAST INDEXED EVIDENCE
    # =========================================================

    def _load_selected_indexed_evidence(
        self,
        *,
        user_id: str,
        document_ids: Sequence[str],
    ) -> list[Dict[str, Any]]:
        """
        Read existing Chroma chunks for the selected user/documents.

        The access path is intentionally discovered defensively so the
        service remains compatible with test fakes and alternate wrappers:

            RetrievalGraph
              -> RetrievalAgent
              -> Retriever
              -> VectorSearch
              -> ChromaManager
        """

        try:
            retrieval_agent = getattr(self.retrieval_graph, "retrieval", None)
            retriever = getattr(retrieval_agent, "retriever", None)
            vector_search = getattr(retriever, "search", None)
            chroma_manager = getattr(vector_search, "chroma", None)

            if chroma_manager is None:
                return []

            collection = chroma_manager.get_collection()
        except Exception:
            return []

        evidence: list[Dict[str, Any]] = []

        for document_id in document_ids:
            try:
                result = collection.get(
                    where={
                        "$and": [
                            {"user_id": str(user_id)},
                            {"document_id": str(document_id)},
                        ]
                    },
                    include=["documents", "metadatas"],
                )
            except Exception:
                return []

            ids = list(result.get("ids") or [])
            documents = list(result.get("documents") or [])
            metadatas = list(result.get("metadatas") or [])

            for index, chunk_id in enumerate(ids):
                content = documents[index] if index < len(documents) else ""
                metadata = metadatas[index] if index < len(metadatas) else {}
                metadata = dict(metadata or {})

                # Reassert requested-scope safety even though Chroma was
                # already filtered by both user and document.
                if str(metadata.get("user_id") or "") != str(user_id):
                    continue
                if str(metadata.get("document_id") or "") != str(document_id):
                    continue

                evidence.append(
                    {
                        "chunk_id": str(chunk_id),
                        "content": str(content or ""),
                        "metadata": metadata,
                        # There is no semantic distance in a direct get.
                        # The comparison agent performs deterministic metric
                        # matching, so a relevance score is not required.
                        "relevance_score": 1.0,
                    }
                )

        return evidence

    # =========================================================
    # FROZEN SEMANTIC FALLBACK
    # =========================================================

    def _semantic_retrieval_fallback(
        self,
        *,
        user_id: str,
        document_ids: Sequence[str],
        metrics: List[str] | None,
        top_k: int,
    ) -> list[Dict[str, Any]]:

        query = self._build_query(metrics=metrics)
        evidence: list[Dict[str, Any]] = []

        # Preserve the baseline per-document retrieval design. A modestly
        # larger comparison-only budget helps alternate retrieval adapters
        # expose complete tables without affecting Chat & Ask.
        per_document_top_k = max(10, min(40, int(top_k)))

        for document_id in document_ids:
            retrieval = self.retrieval_graph.invoke(
                query=query,
                user_id=user_id,
                document_ids=[str(document_id)],
                top_k=per_document_top_k,
            )
            evidence.extend(list(retrieval.get("evidence", []) or []))

        return evidence

    # =========================================================
    # BUILD QUERY
    # =========================================================

    @staticmethod
    def _build_query(
        *,
        metrics: List[str] | None,
    ) -> str:

        if metrics:
            metric_text = ", ".join(str(metric) for metric in metrics)
            return (
                "Find the exact values needed to compare these metrics in "
                f"this document: {metric_text}. Prioritize directly stated "
                "text values, complete tables, spreadsheet rows, totals, "
                "monthly/quarterly periods, and numerical values associated "
                "with these metrics."
            )

        return (
            "Find important numerical metrics in this document for "
            "cross-document comparison. Prioritize Revenue, Profit, "
            "Employees, units sold, prices, complete tables, spreadsheet "
            "rows, totals, and other directly stated metrics."
        )

    # =========================================================
    # COVERAGE
    # =========================================================

    @staticmethod
    def _covers_all_documents(
        *,
        evidence: Sequence[Dict[str, Any]],
        document_ids: Sequence[str],
    ) -> bool:
        found = set()
        for item in evidence:
            metadata = dict(item.get("metadata") or {})
            document_id = metadata.get("document_id")
            if document_id is not None:
                found.add(str(document_id))
        return all(str(document_id) in found for document_id in document_ids)

    # =========================================================
    # DEDUPLICATION
    # =========================================================

    @staticmethod
    def _deduplicate_evidence(
        evidence: Sequence[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:

        result: list[Dict[str, Any]] = []
        seen: set[str] = set()

        for index, item in enumerate(evidence):
            metadata = dict(item.get("metadata") or {})
            chunk_id = str(
                item.get("chunk_id")
                or item.get("postgres_chunk_id")
                or (
                    f"{metadata.get('document_id','')}:"
                    f"{metadata.get('chunk_index','')}:"
                    f"{index}"
                )
            )
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            result.append(item)

        return result

    # =========================================================
    # ORDER EVIDENCE
    # =========================================================

    @staticmethod
    def _order_evidence(
        *,
        evidence: Sequence[Dict[str, Any]],
        document_ids: Sequence[str],
    ) -> list[Dict[str, Any]]:

        order = {
            str(document_id): index
            for index, document_id in enumerate(document_ids)
        }

        def sort_key(item: Dict[str, Any]):
            metadata = dict(item.get("metadata") or {})
            document_id = str(metadata.get("document_id", ""))

            try:
                chunk_index = int(metadata.get("chunk_index", 10**9))
            except (TypeError, ValueError):
                chunk_index = 10**9

            try:
                relevance = float(item.get("relevance_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                relevance = 0.0

            return (
                order.get(document_id, len(order)),
                chunk_index,
                -relevance,
            )

        return sorted(evidence, key=sort_key)
