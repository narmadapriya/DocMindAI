from __future__ import annotations

from typing import Iterable
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document
from app.rag.chunking import RetrievalChunk


class ChunkPersistenceService:
    """
    PostgreSQL persistence service for retrieval chunks.

    Responsibilities
    ----------------
    1. Receive RetrievalChunk objects produced by the existing
       Phase 7 ModalityAwareChunker.
    2. Convert them into PostgreSQL Chunk rows.
    3. Preserve:
         - document_id
         - chunk_index
         - content
         - chunk_type
         - page_number
    4. Prevent duplicate rows during document reprocessing.
    5. Support document-level chunk deletion/querying.

    Important
    ---------
    Rich retrieval metadata such as:

        user_id
        filename
        file_type
        section
        table_id
        image_id
        sheet_name
        source_location
        visual_description

    remains attached to RetrievalChunk.metadata and will later
    be sent to ChromaDB.

    The existing PostgreSQL chunks table intentionally stores
    only the relational/search-traceability fields already
    defined by the frozen database schema.
    """

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    # =========================================================
    # Persist Retrieval Chunks
    # =========================================================

    def persist_chunks(
        self,
        *,
        document: Document,
        chunks: Iterable[RetrievalChunk],
        replace_existing: bool = True,
    ) -> list[Chunk]:
        """
        Persist retrieval chunks for a document.

        Parameters
        ----------
        document:
            Existing PostgreSQL Document record.

        chunks:
            RetrievalChunk objects produced by
            ModalityAwareChunker.

        replace_existing:
            When True, old chunks belonging to the same
            document are removed first.

            This makes document reprocessing idempotent
            and prevents duplicate chunk rows.
        """

        if document is None:
            raise ValueError(
                "document is required."
            )

        if document.id is None:
            raise ValueError(
                "document must have a PostgreSQL id "
                "before chunks can be persisted."
            )

        retrieval_chunks = list(chunks)

        try:

            # -------------------------------------------------
            # Replace previous chunks when reprocessing.
            # -------------------------------------------------

            if replace_existing:

                (
                    self.db.query(Chunk)
                    .filter(
                        Chunk.document_id
                        == document.id
                    )
                    .delete(
                        synchronize_session=False
                    )
                )

            stored_chunks: list[Chunk] = []

            # -------------------------------------------------
            # Convert RetrievalChunk -> ORM Chunk
            # -------------------------------------------------

            for fallback_index, retrieval_chunk in enumerate(
                retrieval_chunks
            ):

                if retrieval_chunk is None:
                    continue

                content = getattr(
                    retrieval_chunk,
                    "content",
                    None,
                )

                if not content:
                    continue

                metadata = getattr(
                    retrieval_chunk,
                    "metadata",
                    None,
                )

                if metadata is None:
                    continue

                chunk_index = getattr(
                    metadata,
                    "chunk_index",
                    None,
                )

                if chunk_index is None:
                    chunk_index = fallback_index

                chunk_type = (
                    getattr(
                        metadata,
                        "chunk_type",
                        None,
                    )
                    or "text"
                )

                page_number = getattr(
                    metadata,
                    "page_number",
                    None,
                )

                db_chunk = Chunk(
                    document_id=document.id,
                    chunk_index=int(
                        chunk_index
                    ),
                    content=str(content),
                    chunk_type=str(
                        chunk_type
                    ),
                    page_number=page_number,
                )

                self.db.add(
                    db_chunk
                )

                stored_chunks.append(
                    db_chunk
                )

            # -------------------------------------------------
            # Single commit for all chunks.
            # -------------------------------------------------

            self.db.commit()

            # -------------------------------------------------
            # Refresh generated UUIDs/timestamps.
            # -------------------------------------------------

            for db_chunk in stored_chunks:
                self.db.refresh(
                    db_chunk
                )

            return stored_chunks

        except Exception:

            self.db.rollback()

            raise

    # =========================================================
    # Get Document Chunks
    # =========================================================

    def get_document_chunks(
        self,
        document_id: UUID,
    ) -> list[Chunk]:
        """
        Return all PostgreSQL chunks belonging to a document.

        Chunks are returned in deterministic chunk order.
        """

        return (
            self.db.query(Chunk)
            .filter(
                Chunk.document_id
                == document_id
            )
            .order_by(
                Chunk.chunk_index.asc()
            )
            .all()
        )

    # =========================================================
    # Count Document Chunks
    # =========================================================

    def count_document_chunks(
        self,
        document_id: UUID,
    ) -> int:
        """
        Return the number of persisted chunks for a document.
        """

        return (
            self.db.query(Chunk)
            .filter(
                Chunk.document_id
                == document_id
            )
            .count()
        )

    # =========================================================
    # Delete Document Chunks
    # =========================================================

    def delete_document_chunks(
        self,
        document_id: UUID,
    ) -> int:
        """
        Delete all PostgreSQL chunks belonging to a document.

        Returns the number of rows deleted.
        """

        try:

            deleted = (
                self.db.query(Chunk)
                .filter(
                    Chunk.document_id
                    == document_id
                )
                .delete(
                    synchronize_session=False
                )
            )

            self.db.commit()

            return int(deleted)

        except Exception:

            self.db.rollback()

            raise