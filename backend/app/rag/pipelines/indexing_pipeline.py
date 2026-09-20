from __future__ import annotations

from typing import Any, Dict, List, Sequence
import time

from app.core.logging import get_logger, log_event
from app.core.performance import EMBEDDING_BATCH_SIZE

logger = get_logger(__name__)

from app.rag.embeddings import (
    EmbeddingModel,
)
from app.rag.vectordb import (
    ChromaManager,
)


class IndexingPipeline:
    """
    Complete DocMindAI indexing pipeline.

    Phase 7:
        RetrievalChunk

        ↓

    Phase 8:
        Embedding

        ↓

        ChromaDB
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        chroma_manager: ChromaManager | None = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        self.embedding_model = (
            embedding_model
            or EmbeddingModel()
        )

        self.chroma = (
            chroma_manager
            or ChromaManager()
        )

        self.batch_size = batch_size

    # ---------------------------------------------------------
    # INDEX DOCUMENT
    # ---------------------------------------------------------

    def index_document(
        self,
        chunks: Sequence[Any],
        *,
        document_id: str,
        user_id: str,
        reindex: bool = False,
    ) -> Dict[str, Any]:

        started = time.perf_counter()
        if not document_id:
            raise ValueError(
                "document_id is required."
            )

        if not user_id:
            raise ValueError(
                "user_id is required."
            )

        # -----------------------------------------------------
        # Re-index strategy
        # -----------------------------------------------------
        #
        # IMPORTANT FOR READY-STATE STABILITY:
        # Do not delete the currently usable document vectors here.
        # Embedding generation can take several seconds on an 8-GB
        # local machine, especially on the first Ollama request.
        # Existing vectors are kept available until all replacement
        # embeddings have been generated successfully.

        if not chunks:
            return {
                "status": "success",
                "document_id": document_id,
                "user_id": user_id,
                "indexed": 0,
                "skipped": 0,
                "total": 0,
                "reindexed": reindex,
            }

        # -----------------------------------------------------
        # Prepare chunks
        # -----------------------------------------------------

        valid_chunks = []

        for chunk in chunks:

            chunk_id = getattr(
                chunk,
                "chunk_id",
                None,
            )

            content = getattr(
                chunk,
                "content",
                None,
            )

            if not chunk_id or not content:
                continue

            valid_chunks.append(chunk)

        if not valid_chunks:
            return {
                "status": "success",
                "document_id": document_id,
                "user_id": user_id,
                "indexed": 0,
                "skipped": len(chunks),
                "total": len(chunks),
                "reindexed": reindex,
            }

        # -----------------------------------------------------
        # Duplicate prevention
        # -----------------------------------------------------

        ids = [
            chunk.chunk_id
            for chunk in valid_chunks
        ]

        existing = set()

        if not reindex:
            existing = self.chroma.existing_ids(
                ids
            )

        chunks_to_index = [
            chunk
            for chunk in valid_chunks
            if chunk.chunk_id not in existing
        ]

        skipped = len(
            valid_chunks
        ) - len(
            chunks_to_index
        )

        if not chunks_to_index:
            return {
                "status": "success",
                "document_id": document_id,
                "user_id": user_id,
                "indexed": 0,
                "skipped": skipped,
                "total": len(chunks),
                "reindexed": False,
            }

        # -----------------------------------------------------
        # Generate embeddings
        # -----------------------------------------------------

        texts = [
            chunk.content
            for chunk in chunks_to_index
        ]

        embeddings = (
            self.embedding_model.embed_batch(
                texts,
                batch_size=self.batch_size,
            )
        )

        if len(embeddings) != len(chunks_to_index):
            raise RuntimeError(
                "Embedding count does not match chunks selected "
                "for indexing."
            )

        # -----------------------------------------------------
        # Safe replacement point
        # -----------------------------------------------------
        # All potentially slow embedding work has completed. Only
        # now remove the previous vectors. This reduces the period
        # where /api/upload/status could observe vector_count == 0
        # during multimodal re-indexing from seconds/minutes to the
        # tiny delete -> upsert window. Final Chroma contents remain
        # exactly the same as the frozen reindex=True behaviour.

        if reindex:
            self.chroma.delete_by_document(
                document_id
            )

        # -----------------------------------------------------
        # Metadata
        # -----------------------------------------------------

        metadatas = []

        for chunk in chunks_to_index:

            metadata = (
                chunk.metadata.to_dict()
                if hasattr(
                    chunk.metadata,
                    "to_dict",
                )
                else dict(
                    chunk.metadata
                )
            )

            # Enforce isolation metadata.
            metadata[
                "document_id"
            ] = document_id

            metadata[
                "user_id"
            ] = user_id

            metadatas.append(
                metadata
            )

        # -----------------------------------------------------
        # Store in ChromaDB
        # -----------------------------------------------------

        result = self.chroma.upsert(
            ids=[
                chunk.chunk_id
                for chunk in chunks_to_index
            ],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        response = {
            "status": "success",
            "document_id": document_id,
            "user_id": user_id,
            "indexed": len(chunks_to_index),
            "skipped": skipped,
            "total": len(chunks),
            "reindexed": reindex,
            "chroma": result,
        }
        log_event(logger, "indexing", document_id=document_id, indexed=response["indexed"], skipped=skipped, latency_ms=round((time.perf_counter()-started)*1000, 2))
        return response

    # ---------------------------------------------------------
    # DELETE
    # ---------------------------------------------------------

    def delete_document(
        self,
        document_id: str,
    ) -> Dict[str, Any]:

        return self.chroma.delete_by_document(
            document_id
        )

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    def document_status(
        self,
        document_id: str,
    ) -> Dict[str, Any]:

        count = self.chroma.document_count(
            document_id
        )

        return {
            "document_id": document_id,
            "indexed": count > 0,
            "chunk_count": count,
            "status": (
                "indexed"
                if count > 0
                else "not_indexed"
            ),
        }

    def user_status(
        self,
        user_id: str,
    ) -> Dict[str, Any]:

        count = self.chroma.user_count(
            user_id
        )

        return {
            "user_id": user_id,
            "indexed": count > 0,
            "chunk_count": count,
            "status": (
                "indexed"
                if count > 0
                else "not_indexed"
            ),
        }

    def system_status(self) -> Dict[str, Any]:

        return self.chroma.status()