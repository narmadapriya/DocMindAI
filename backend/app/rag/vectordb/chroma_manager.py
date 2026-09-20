from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence


class ChromaDBError(RuntimeError):
    """Raised when a ChromaDB operation fails."""


class ChromaManager:
    """
    Persistent ChromaDB manager for DocMindAI.

    Phase 9 responsibilities:
        - collection management
        - insertion / upsert
        - duplicate prevention
        - user isolation
        - document filtering
        - metadata filtering
        - semantic querying
        - deletion
        - status information
    """

    COLLECTION_NAME = "docmindai_documents"

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ):
        self.persist_directory = (
            persist_directory
            or os.getenv(
                "CHROMA_DB_DIR",
                "app/chroma_db",
            )
        )

        self.collection_name = (
            collection_name
            or self.COLLECTION_NAME
        )

        self._client = None
        self._collection = None

    # =========================================================
    # CONNECTION
    # =========================================================

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb

                os.makedirs(
                    self.persist_directory,
                    exist_ok=True,
                )

                self._client = chromadb.PersistentClient(
                    path=self.persist_directory
                )

            except ImportError as exc:
                raise ChromaDBError(
                    "The 'chromadb' package is not installed."
                ) from exc

        return self._client

    # =========================================================
    # COLLECTION
    # =========================================================

    def get_collection(self):
        """
        Return the configured Chroma collection.

        The collection uses cosine distance because the
        Phase 9 relevance calculation assumes:

            relevance = 1 - cosine_distance
        """

        if self._collection is None:
            client = self._get_client()

            self._collection = (
                client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={
                        "project": "DocMindAI",
                        "purpose": "multimodal-rag",
                        "hnsw:space": "cosine",
                    },
                )
            )

        return self._collection

    # =========================================================
    # ADD / UPSERT
    # =========================================================

    def upsert(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not (
            len(ids)
            == len(embeddings)
            == len(documents)
            == len(metadatas)
        ):
            raise ValueError(
                "ids, embeddings, documents and metadatas "
                "must have equal lengths."
            )

        if not ids:
            return {
                "status": "success",
                "inserted": 0,
                "updated": 0,
            }

        collection = self.get_collection()

        collection.upsert(
            ids=list(ids),
            embeddings=[
                list(vector)
                for vector in embeddings
            ],
            documents=list(documents),
            metadatas=[
                self._sanitize_metadata(metadata)
                for metadata in metadatas
            ],
        )

        return {
            "status": "success",
            "count": len(ids),
        }

    # =========================================================
    # DUPLICATE CHECK
    # =========================================================

    def existing_ids(
        self,
        ids: Sequence[str],
    ) -> set[str]:

        if not ids:
            return set()

        collection = self.get_collection()

        result = collection.get(
            ids=list(ids),
            include=[],
        )

        return set(
            result.get("ids", [])
        )

    # =========================================================
    # GET BY IDS
    # =========================================================

    def get_by_ids(
        self,
        ids: Sequence[str],
    ) -> Dict[str, Any]:

        if not ids:
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
            }

        return self.get_collection().get(
            ids=list(ids),
            include=[
                "documents",
                "metadatas",
                "embeddings",
            ],
        )

    # =========================================================
    # QUERY
    # =========================================================

    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        if not query_embeddings:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        if n_results <= 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        collection = self.get_collection()

        # -----------------------------------------------------
        # Do not send an empty where clause to Chroma.
        # -----------------------------------------------------

        kwargs: Dict[str, Any] = {
            "query_embeddings": [
                list(vector)
                for vector in query_embeddings
            ],
            "n_results": int(n_results),
        }

        if where:
            kwargs["where"] = where

        try:
            return collection.query(**kwargs)

        except Exception as exc:
            raise ChromaDBError(
                f"ChromaDB query failed: {exc}"
            ) from exc

    # =========================================================
    # USER / DOCUMENT FILTERING
    # =========================================================

    def query_by_document(
        self,
        *,
        query_embedding: Sequence[float],
        document_id: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:

        return self.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=n_results,
            where={
                "document_id": document_id
            },
        )

    def query_by_user(
        self,
        *,
        query_embedding: Sequence[float],
        user_id: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:

        return self.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=n_results,
            where={
                "user_id": user_id
            },
        )

    # =========================================================
    # DELETE
    # =========================================================

    def delete_by_document(
        self,
        document_id: str,
    ) -> Dict[str, Any]:

        collection = self.get_collection()

        existing = collection.get(
            where={
                "document_id": document_id
            },
            include=[],
        )

        ids = existing.get(
            "ids",
            [],
        )

        if ids:
            collection.delete(
                ids=ids
            )

        return {
            "status": "success",
            "document_id": document_id,
            "deleted": len(ids),
        }

    def delete_by_user(
        self,
        user_id: str,
    ) -> Dict[str, Any]:

        collection = self.get_collection()

        existing = collection.get(
            where={
                "user_id": user_id
            },
            include=[],
        )

        ids = existing.get(
            "ids",
            [],
        )

        if ids:
            collection.delete(
                ids=ids
            )

        return {
            "status": "success",
            "user_id": user_id,
            "deleted": len(ids),
        }

    # =========================================================
    # COUNT / STATUS
    # =========================================================

    def count(self) -> int:
        return self.get_collection().count()

    def document_count(
        self,
        document_id: str,
    ) -> int:

        result = self.get_collection().get(
            where={
                "document_id": document_id
            },
            include=[],
        )

        return len(
            result.get("ids", [])
        )

    def user_count(
        self,
        user_id: str,
    ) -> int:

        result = self.get_collection().get(
            where={
                "user_id": user_id
            },
            include=[],
        )

        return len(
            result.get("ids", [])
        )

    def status(self) -> Dict[str, Any]:
        return {
            "status": "ready",
            "collection": self.collection_name,
            "persist_directory": (
                self.persist_directory
            ),
            "count": self.count(),
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset_collection(self) -> None:

        client = self._get_client()

        try:
            client.delete_collection(
                self.collection_name
            )
        except Exception:
            pass

        self._collection = None

    # =========================================================
    # METADATA
    # =========================================================

    @staticmethod
    def _sanitize_metadata(
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:

        result: Dict[str, Any] = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            ):
                result[key] = value

            else:
                result[key] = str(value)

        return result