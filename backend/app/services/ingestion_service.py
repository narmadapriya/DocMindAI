from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time
from typing import Any, Callable
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document
from app.models.user import User

from app.rag.chunking import (
    ModalityAwareChunker,
    RetrievalChunk,
)

from app.rag.pipelines.indexing_pipeline import (
    IndexingPipeline,
)

from app.schemas.parsed_document import (
    ParsedDocument,
)

from app.services.chunk_service import (
    ChunkPersistenceService,
)

from app.services.document_service import (
    DocumentService,
)

from app.services.multimodal_processing_service import (
    MultimodalProcessingService,
)

from app.services.parser_service import (
    ParserService,
)
from app.core.logging import get_logger, log_event

logger = get_logger(__name__)


# ============================================================
# Ingestion Result
# ============================================================

@dataclass
class IngestionResult:
    """
    Complete DocMindAI ingestion result.

    Step 1:
        document

    Steps 2-3:
        parsed_document

    Step 6:
        multimodal enrichment

    Step 4:
        retrieval_chunks
        stored_chunks

    Step 5:
        indexing_result
    """

    document: Document

    parsed_document: ParsedDocument

    retrieval_chunks: list[
        RetrievalChunk
    ] = field(
        default_factory=list
    )

    stored_chunks: list[
        Chunk
    ] = field(
        default_factory=list
    )

    indexing_result: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    # ========================================================
    # Step 4
    # ========================================================

    @property
    def chunk_count(self) -> int:
        """
        Number of PostgreSQL chunks persisted.
        """

        return len(
            self.stored_chunks
        )

    # ========================================================
    # Step 5
    # ========================================================

    @property
    def indexed_count(self) -> int:
        """
        Number of chunks newly indexed into ChromaDB.
        """

        return int(
            self.indexing_result.get(
                "indexed",
                0,
            )
        )

    @property
    def skipped_index_count(self) -> int:
        """
        Number of existing ChromaDB vectors skipped.
        """

        return int(
            self.indexing_result.get(
                "skipped",
                0,
            )
        )

    @property
    def indexed(self) -> bool:
        """
        True when ChromaDB indexing completed successfully.
        """

        return (
            self.indexing_result.get(
                "status"
            )
            == "success"
        )

    # ========================================================
    # Step 6
    # ========================================================

    @property
    def multimodal_processed(self) -> bool:
        """
        Whether multimodal enrichment completed.
        """

        return bool(
            self.parsed_document
            .metadata
            .get(
                "multimodal_processed",
                False,
            )
        )

    @property
    def multimodal_evidence_count(self) -> int:
        """
        Number of multimodal evidence items recorded.
        """

        return int(
            self.parsed_document
            .metadata
            .get(
                "multimodal_evidence_count",
                0,
            )
        )


# ============================================================
# Document Ingestion Service
# ============================================================

class DocumentIngestionService:
    """
    Complete DocMindAI Document Ingestion / RAG pipeline.

    ----------------------------------------------------------
    SYNCHRONOUS PIPELINE
    ----------------------------------------------------------

        UploadFile
            ↓
        DocumentService
            ↓
        PostgreSQL Document
            ↓
        ParserService
            ↓
        Unified ParsedDocument
            ↓
        MultimodalProcessingService
            ↓
        Enriched ParsedDocument
            ↓
        ModalityAwareChunker
            ↓
        RetrievalChunk[]
            ↓
        ChunkPersistenceService
            ↓
        PostgreSQL chunks
            ↓
        IndexingPipeline
            ↓
        Embeddings
            ↓
        ChromaDB

    ----------------------------------------------------------
    OPTIMIZED BACKGROUND PIPELINE
    ----------------------------------------------------------

        Existing PostgreSQL Document
            ↓
        Parse
            ↓
        FAST:
            chunk
            PostgreSQL
            embeddings
            ChromaDB
            ↓
        READY FOR RAG
            ↓
        SLOW BACKGROUND:
            OCR
            Qwen2.5-VL
            multimodal enrichment
            rechunk
            PostgreSQL replacement
            ChromaDB replacement

    Supported formats:

        PDF
        DOCX
        TXT
        CSV
        XLSX
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        db: Session,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        indexing_pipeline: (
            IndexingPipeline
            | None
        ) = None,
        multimodal_service: (
            MultimodalProcessingService
            | None
        ) = None,
    ):

        self.db = db

        # ====================================================
        # STEP 1
        # ====================================================

        self.document_service = (
            DocumentService(
                db
            )
        )

        # ====================================================
        # STEPS 2-3
        # ====================================================

        self.parser_service = (
            ParserService()
        )

        # ====================================================
        # STEP 6
        # ====================================================

        self.multimodal_service = (
            multimodal_service
            or MultimodalProcessingService()
        )

        # ====================================================
        # STEP 4
        # ====================================================

        self.chunker = (
            ModalityAwareChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

        self.chunk_service = (
            ChunkPersistenceService(
                db
            )
        )

        # ====================================================
        # STEP 5
        # ====================================================

        self.indexing_pipeline = (
            indexing_pipeline
            or IndexingPipeline()
        )

    # ========================================================
    # INGEST NEW DOCUMENT — SYNCHRONOUS
    # ========================================================

    def ingest_document(
        self,
        file: UploadFile,
        owner: User,
    ) -> IngestionResult:
        """
        Execute the complete synchronous ingestion pipeline.

        IMPORTANT:
        Keep this method synchronous because existing frozen
        Steps 4-6 integration tests depend on the complete
        pipeline finishing before this method returns.

        The optimized FastAPI upload endpoint should use:

            DocumentService.upload_document()
                    ↓
            background task
                    ↓
            process_existing_document()

        instead of calling this method directly.
        """

        # ====================================================
        # STEP 1
        # Upload + PostgreSQL Document
        # ====================================================

        document = (
            self.document_service
            .upload_document(
                file=file,
                owner=owner,
            )
        )

        # ====================================================
        # Complete processing
        # ====================================================

        (
            parsed_document,
            retrieval_chunks,
            stored_chunks,
            indexing_result,
        ) = self._process_document_complete(
            document=document,
            owner=owner,
            reindex=False,
        )

        # ====================================================
        # Diagnostics
        # ====================================================

        print(
            f"[INGESTION] SUCCESS: "
            f"{document.original_filename}"
        )

        print(
            f"[INGESTION] Document ID: "
            f"{document.id}"
        )

        print(
            f"[MULTIMODAL] Processed: "
            f"{parsed_document.metadata.get(
                'multimodal_processed',
                False
            )}"
        )

        print(
            f"[MULTIMODAL] Evidence: "
            f"{parsed_document.metadata.get(
                'multimodal_evidence_count',
                0
            )}"
        )

        print(
            f"[CHUNKING] Generated: "
            f"{len(retrieval_chunks)}"
        )

        print(
            f"[POSTGRESQL] Persisted: "
            f"{len(stored_chunks)}"
        )

        print(
            f"[CHROMADB] Indexed: "
            f"{indexing_result.get(
                'indexed',
                0
            )}"
        )

        print(
            f"[CHROMADB] Skipped: "
            f"{indexing_result.get(
                'skipped',
                0
            )}"
        )

        # ====================================================
        # Return
        # ====================================================

        return IngestionResult(
            document=document,
            parsed_document=(
                parsed_document
            ),
            retrieval_chunks=(
                retrieval_chunks
            ),
            stored_chunks=(
                stored_chunks
            ),
            indexing_result=(
                indexing_result
            ),
        )

    # ========================================================
    # INGEST MULTIPLE DOCUMENTS — SYNCHRONOUS
    # ========================================================

    def ingest_multiple_documents(
        self,
        files: list[
            UploadFile
        ],
        owner: User,
    ) -> list[
        IngestionResult
    ]:
        """
        Complete synchronous ingestion for multiple files.

        Preserved for:
            tests
            CLI
            internal workflows

        The optimized HTTP /multiple endpoint should save the
        files first and then schedule background processing.
        """

        results: list[
            IngestionResult
        ] = []

        for file in files:

            result = (
                self.ingest_document(
                    file=file,
                    owner=owner,
                )
            )

            results.append(
                result
            )

        return results

    # ========================================================
    # INGEST EXISTING DOCUMENT — COMPLETE SYNCHRONOUS
    # ========================================================

    def ingest_existing_document(
        self,
        document: Document,
        owner: User,
    ) -> ParsedDocument:
        """
        Completely reprocess an existing document.

        Existing frozen return contract:
            ParsedDocument

        This performs full multimodal processing before
        replacing PostgreSQL/Chroma data.
        """

        # ====================================================
        # Ownership
        # ====================================================

        self._validate_owner(
            document=document,
            owner=owner,
        )

        # ====================================================
        # Complete processing
        # ====================================================

        (
            parsed_document,
            retrieval_chunks,
            stored_chunks,
            indexing_result,
        ) = self._process_document_complete(
            document=document,
            owner=owner,
            reindex=True,
        )

        # ====================================================
        # Diagnostics
        # ====================================================

        print(
            f"[REPROCESS] SUCCESS: "
            f"{document.original_filename}"
        )

        print(
            f"[REPROCESS] Document ID: "
            f"{document.id}"
        )

        print(
            f"[REPROCESS] Retrieval chunks: "
            f"{len(retrieval_chunks)}"
        )

        print(
            f"[REPROCESS] PostgreSQL chunks: "
            f"{len(stored_chunks)}"
        )

        print(
            f"[REPROCESS] ChromaDB indexed: "
            f"{indexing_result.get(
                'indexed',
                0
            )}"
        )

        # Preserve frozen Step 3 contract.
        return parsed_document

    # ========================================================
    # OPTIMIZED BACKGROUND ENTRY POINT
    # ========================================================

    def process_existing_document(
        self,
        *,
        document_id: UUID | str,
        owner_id: UUID | str,
        on_fast_ready: Callable[[], None] | None = None,
        post_fast_ready_delay_seconds: float = 0.0,
    ) -> ParsedDocument:
        """
        Optimized background processing.

        ------------------------------------------------------
        STAGE 1 — FAST
        ------------------------------------------------------

            Parse
              ↓
            Chunk
              ↓
            PostgreSQL
              ↓
            Embeddings
              ↓
            ChromaDB
              ↓
            READY FOR RAG

        ------------------------------------------------------
        STAGE 2 — SLOW MULTIMODAL ENRICHMENT
        ------------------------------------------------------

            OCR
              ↓
            Qwen2.5-VL
              ↓
            enriched chunks
              ↓
            PostgreSQL replacement
              ↓
            ChromaDB replacement

        This allows /api/upload/status/{document_id} to become
        ready much sooner on an 8 GB local machine.
        """

        # ====================================================
        # Reload Document using background DB session
        # ====================================================

        document = (
            self.db.query(
                Document
            )
            .filter(
                Document.id
                == document_id
            )
            .first()
        )

        if document is None:

            raise ValueError(
                f"Document not found: "
                f"{document_id}"
            )

        # ====================================================
        # Reload User using background DB session
        # ====================================================

        owner = (
            self.db.query(
                User
            )
            .filter(
                User.id
                == owner_id
            )
            .first()
        )

        if owner is None:

            raise ValueError(
                f"User not found: "
                f"{owner_id}"
            )

        # ====================================================
        # Ownership
        # ====================================================

        self._validate_owner(
            document=document,
            owner=owner,
        )

        print(
            "[BACKGROUND] Starting "
            "optimized ingestion."
        )

        print(
            f"[BACKGROUND] Document ID: "
            f"{document.id}"
        )

        print(
            f"[BACKGROUND] File: "
            f"{document.original_filename}"
        )

        # ====================================================
        # PARSE ONCE
        # ====================================================

        parsed_document = (
            self.parser_service
            .parse_document(
                file_path=(
                    document.file_path
                ),
                document_id=(
                    document.id
                ),
                filename=(
                    document.original_filename
                ),
            )
        )

        if parsed_document is None:

            raise RuntimeError(
                "ParserService returned "
                "no ParsedDocument."
            )

        # ====================================================
        # STAGE 1
        # FAST RETRIEVAL INDEX
        # ====================================================

        print(
            "[BACKGROUND][FAST] "
            "Generating initial chunks..."
        )

        initial_chunks = (
            self.chunker
            .chunk_document(
                parsed_document,
                document_id=str(
                    document.id
                ),
                user_id=str(
                    owner.id
                ),
            )
        )

        if not initial_chunks:

            raise RuntimeError(
                "Fast ingestion produced "
                "zero retrieval chunks."
            )

        # ----------------------------------------------------
        # FAST READY INDEX
        # ----------------------------------------------------
        # Text and structured table evidence are already complete
        # immediately after parsing. Image/chart chunks at this point
        # are only pre-enrichment placeholders and will be replaced by
        # Stage 2 after OCR/Qwen processing.
        #
        # Indexing only ready evidence here reduces the number of
        # embeddings required before ready_for_rag=True while preserving
        # the final enriched RAG index produced by Stage 2.

        fast_chunks = self._select_fast_ready_chunks(
            initial_chunks
        )

        # A visual-only document must still remain ingestible. In that
        # uncommon case preserve the original initial chunks.
        if not fast_chunks:
            fast_chunks = initial_chunks

        print(
            f"[BACKGROUND][FAST] "
            f"Initial chunks: "
            f"{len(initial_chunks)}"
        )

        print(
            f"[BACKGROUND][FAST] "
            f"Ready-index chunks: "
            f"{len(fast_chunks)}"
        )

        # ====================================================
        # STAGE 1A
        # PostgreSQL persistence
        # ====================================================

        fast_stored_chunks = (
            self.chunk_service
            .persist_chunks(
                document=document,
                chunks=fast_chunks,
                replace_existing=True,
            )
        )

        if not fast_stored_chunks:

            raise RuntimeError(
                "Fast ingestion produced "
                "zero PostgreSQL chunks."
            )

        if (
            len(fast_stored_chunks)
            != len(fast_chunks)
        ):

            raise RuntimeError(
                "Fast PostgreSQL persistence "
                "count mismatch. "
                f"Generated={len(fast_chunks)}, "
                f"Persisted={len(fast_stored_chunks)}"
            )

        print(
            f"[BACKGROUND][FAST] "
            f"PostgreSQL chunks: "
            f"{len(fast_stored_chunks)}"
        )

        # ====================================================
        # STAGE 1B
        # ChromaDB
        # ====================================================

        fast_index_result = (
            self.indexing_pipeline
            .index_document(
                fast_chunks,
                document_id=str(
                    document.id
                ),
                user_id=str(
                    owner.id
                ),
                reindex=True,
            )
        )

        self._validate_indexing_result(
            indexing_result=(
                fast_index_result
            ),
            stage="fast",
        )

        print(
            f"[BACKGROUND][FAST] "
            f"Chroma indexed: "
            f"{fast_index_result.get(
                'indexed',
                0
            )}"
        )

        print(
            f"[BACKGROUND][FAST] "
            f"Chroma skipped: "
            f"{fast_index_result.get(
                'skipped',
                0
            )}"
        )

        print(
            "[BACKGROUND][FAST] "
            "DOCUMENT IS READY FOR RAG."
        )

        # ----------------------------------------------------
        # Notify the background coordinator immediately after
        # the usable Stage-1 index exists.  This keeps the
        # Processing -> Ready critical path independent from
        # OCR/Qwen2.5-VL visual enrichment.  The callback is
        # optional, so all existing synchronous/internal callers
        # retain the frozen behaviour.
        # ----------------------------------------------------
        if on_fast_ready is not None:
            try:
                on_fast_ready()
            except Exception as exc:
                # Readiness notification is operational metadata
                # only.  It must never invalidate a successfully
                # built Stage-1 RAG index.
                print(
                    "[BACKGROUND][FAST] "
                    "Ready callback failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        # ====================================================
        # Determine if slow visual stage is necessary
        # ====================================================

        has_images = bool(
            parsed_document.images
        )

        has_charts = bool(
            parsed_document.charts
        )

        has_visual_assets = (
            has_images
            or has_charts
        )

        # ----------------------------------------------------
        # Tables are already structured by parser and already
        # available in Stage 1.
        # ----------------------------------------------------

        if not has_visual_assets:

            parsed_document.metadata[
                "multimodal_processed"
            ] = True

            parsed_document.metadata[
                "multimodal_evidence_count"
            ] = len(
                parsed_document.tables
            )

            parsed_document.metadata[
                "multimodal_summary"
            ] = {
                "tables":
                    len(
                        parsed_document.tables
                    ),

                "images":
                    0,

                "charts":
                    0,
            }

            print(
                "[BACKGROUND][MULTIMODAL] "
                "No image/chart assets found."
            )

            print(
                "[BACKGROUND][MULTIMODAL] "
                "Vision inference skipped."
            )

            return parsed_document

        # ====================================================
        # STAGE 2
        # EXPENSIVE MULTIMODAL PROCESSING
        # ====================================================

        # The Stage-1 index is already usable and includes unique
        # raw visual asset chunks, so a visual question can still
        # attach the original image at reasoning time.  Give the
        # newly-ready document a short foreground window before
        # starting heavy OCR/Qwen enrichment.  This prevents the
        # first Chat/Summary/Comparison request from competing with
        # local vision inference on an 8-GB machine.
        delay_seconds = max(
            0.0,
            float(post_fast_ready_delay_seconds or 0.0),
        )

        if delay_seconds > 0:
            print(
                "[BACKGROUND][MULTIMODAL] "
                f"Deferred for {delay_seconds:.1f}s "
                "after fast-ready."
            )
            time.sleep(delay_seconds)

        print(
            "[BACKGROUND][MULTIMODAL] "
            "Starting OCR/Qwen2.5-VL..."
        )

        try:

            enriched_document = (
                self.multimodal_service
                .process_document(
                    parsed_document
                )
            )

        except Exception as exc:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Stage 1 has already produced a usable RAG index.
            # A vision error must not destroy that index.
            # ------------------------------------------------

            print(
                "[BACKGROUND][MULTIMODAL] "
                "Visual enrichment failed."
            )

            print(
                "[BACKGROUND][MULTIMODAL] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                "[BACKGROUND] "
                "Keeping Stage 1 RAG index."
            )

            return parsed_document

        if enriched_document is None:

            print(
                "[BACKGROUND][MULTIMODAL] "
                "No enriched document returned."
            )

            print(
                "[BACKGROUND] "
                "Keeping Stage 1 RAG index."
            )

            return parsed_document

        # ====================================================
        # Generate enriched chunks
        # ====================================================

        enriched_chunks = (
            self.chunker
            .chunk_document(
                enriched_document,
                document_id=str(
                    document.id
                ),
                user_id=str(
                    owner.id
                ),
            )
        )

        if not enriched_chunks:

            print(
                "[BACKGROUND][MULTIMODAL] "
                "Enrichment generated zero chunks."
            )

            print(
                "[BACKGROUND] "
                "Keeping Stage 1 RAG index."
            )

            return parsed_document

        print(
            f"[BACKGROUND][MULTIMODAL] "
            f"Enriched chunks: "
            f"{len(enriched_chunks)}"
        )

        # ====================================================
        # IMPORTANT:
        #
        # Index enriched Chroma representation FIRST.
        #
        # If this fails, keep the original PostgreSQL Stage 1
        # chunks instead of replacing them with data whose
        # vector representation failed.
        # ====================================================

        try:

            enriched_index_result = (
                self.indexing_pipeline
                .index_document(
                    enriched_chunks,
                    document_id=str(
                        document.id
                    ),
                    user_id=str(
                        owner.id
                    ),
                    reindex=True,
                )
            )

            self._validate_indexing_result(
                indexing_result=(
                    enriched_index_result
                ),
                stage="multimodal",
            )

        except Exception as exc:

            print(
                "[BACKGROUND][MULTIMODAL] "
                "Enriched Chroma indexing failed."
            )

            print(
                "[BACKGROUND][MULTIMODAL] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            # ------------------------------------------------
            # Restore Stage 1 Chroma index because reindex=True
            # may already have deleted previous vectors.
            # ------------------------------------------------

            print(
                "[BACKGROUND][MULTIMODAL] "
                "Restoring Stage 1 vectors..."
            )

            restore_result = (
                self.indexing_pipeline
                .index_document(
                    fast_chunks,
                    document_id=str(
                        document.id
                    ),
                    user_id=str(
                        owner.id
                    ),
                    reindex=True,
                )
            )

            self._validate_indexing_result(
                indexing_result=(
                    restore_result
                ),
                stage="restore",
            )

            print(
                "[BACKGROUND] "
                "Stage 1 RAG index restored."
            )

            return parsed_document

        # ====================================================
        # Replace PostgreSQL only AFTER successful vector index
        # ====================================================

        try:

            enriched_stored_chunks = (
                self.chunk_service
                .persist_chunks(
                    document=document,
                    chunks=enriched_chunks,
                    replace_existing=True,
                )
            )

            if not enriched_stored_chunks:

                raise RuntimeError(
                    "Enriched PostgreSQL "
                    "persistence produced "
                    "zero chunks."
                )

            if (
                len(enriched_stored_chunks)
                != len(enriched_chunks)
            ):

                raise RuntimeError(
                    "Enriched PostgreSQL chunk "
                    "count mismatch. "
                    f"Generated={len(enriched_chunks)}, "
                    f"Persisted={len(enriched_stored_chunks)}"
                )

        except Exception as exc:

            print(
                "[BACKGROUND][MULTIMODAL] "
                "PostgreSQL replacement failed."
            )

            print(
                "[BACKGROUND][MULTIMODAL] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            # ------------------------------------------------
            # Restore Chroma to the same Stage 1 data that
            # PostgreSQL still contains.
            # ------------------------------------------------

            print(
                "[BACKGROUND][MULTIMODAL] "
                "Restoring Stage 1 vectors..."
            )

            restore_result = (
                self.indexing_pipeline
                .index_document(
                    fast_chunks,
                    document_id=str(
                        document.id
                    ),
                    user_id=str(
                        owner.id
                    ),
                    reindex=True,
                )
            )

            self._validate_indexing_result(
                indexing_result=(
                    restore_result
                ),
                stage="restore",
            )

            return parsed_document

        # ====================================================
        # Multimodal enrichment complete
        # ====================================================

        print(
            "[BACKGROUND][MULTIMODAL] "
            "Enrichment completed successfully."
        )

        print(
            f"[BACKGROUND][MULTIMODAL] "
            f"PostgreSQL chunks: "
            f"{len(enriched_stored_chunks)}"
        )

        print(
            f"[BACKGROUND][MULTIMODAL] "
            f"Chroma indexed: "
            f"{enriched_index_result.get(
                'indexed',
                0
            )}"
        )

        return enriched_document

    # ========================================================
    # FAST-READY CHUNK SELECTION
    # ========================================================

    @staticmethod
    def _select_fast_ready_chunks(
        initial_chunks: list[RetrievalChunk],
    ) -> list[RetrievalChunk]:
        """
        Select evidence that is immediately usable before visual
        enrichment finishes.

        Text and structured tables are always kept.  Raw image chunks
        are also kept so visual Chat & Ask can attach the original
        asset even while enrichment is deferred.  Repeated embedded
        images (for example a logo on every PDF page) are deduplicated
        by file-content hash so they do not consume extra embeddings.

        Chart placeholders are intentionally skipped at this stage for
        PDFs because each PDF chart candidate is backed by the same
        image that is already represented by an image chunk.  Enriched
        chart chunks are added by the unchanged multimodal stage later.
        """

        ready: list[RetrievalChunk] = []
        seen_visual_assets: set[str] = set()

        for chunk in initial_chunks:
            metadata = getattr(
                chunk,
                "metadata",
                None,
            )

            chunk_type = str(
                getattr(
                    metadata,
                    "chunk_type",
                    "",
                )
                or ""
            ).strip().lower()

            content = str(
                getattr(
                    chunk,
                    "content",
                    "",
                )
                or ""
            ).strip()

            if not content:
                continue

            if chunk_type in {
                "text",
                "table",
            }:
                ready.append(chunk)
                continue

            if chunk_type != "image":
                continue

            source_location = getattr(
                metadata,
                "source_location",
                None,
            )

            visual_key = str(
                source_location
                or getattr(
                    metadata,
                    "image_id",
                    "",
                )
                or chunk.chunk_id
            )

            if source_location:
                path = Path(
                    str(source_location)
                )

                if path.exists() and path.is_file():
                    try:
                        digest = hashlib.sha1()
                        with path.open("rb") as handle:
                            while True:
                                block = handle.read(1024 * 1024)
                                if not block:
                                    break
                                digest.update(block)
                        visual_key = digest.hexdigest()
                    except OSError:
                        # Path identity is a safe fallback.
                        visual_key = str(path.resolve())

            if visual_key in seen_visual_assets:
                continue

            seen_visual_assets.add(visual_key)
            ready.append(chunk)

        return ready


    # ========================================================
    # COMPLETE PROCESSING PIPELINE
    # ========================================================

    def _process_document_complete(
        self,
        *,
        document: Document,
        owner: User,
        reindex: bool,
    ) -> tuple[
        ParsedDocument,
        list[
            RetrievalChunk
        ],
        list[
            Chunk
        ],
        dict[
            str,
            Any,
        ],
    ]:
        """
        Original complete ingestion pipeline.

        Used by:
            ingest_document()
            ingest_existing_document()

        This intentionally keeps multimodal processing BEFORE
        chunking so frozen Step 4-6 behavior remains unchanged.
        """

        # ====================================================
        # Ownership
        # ====================================================

        self._validate_owner(
            document=document,
            owner=owner,
        )

        # ====================================================
        # STEPS 2-3
        # Parse
        # ====================================================

        parsed_document = (
            self.parser_service
            .parse_document(
                file_path=(
                    document.file_path
                ),
                document_id=(
                    document.id
                ),
                filename=(
                    document.original_filename
                ),
            )
        )

        if parsed_document is None:

            raise RuntimeError(
                "ParserService returned "
                "no ParsedDocument."
            )

        # ====================================================
        # STEP 6
        # Multimodal enrichment
        # ====================================================

        parsed_document = (
            self.multimodal_service
            .process_document(
                parsed_document
            )
        )

        if parsed_document is None:

            raise RuntimeError(
                "MultimodalProcessingService "
                "returned no ParsedDocument."
            )

        # ====================================================
        # STEP 4A
        # Chunking
        # ====================================================

        chunk_started = time.perf_counter()
        retrieval_chunks = (
            self.chunker
            .chunk_document(
                parsed_document,
                document_id=str(
                    document.id
                ),
                user_id=str(
                    owner.id
                ),
            )
        )
        log_event(
            logger,
            "chunking",
            document_id=str(document.id),
            user_id=str(owner.id),
            chunk_count=len(retrieval_chunks or []),
            latency_ms=round((time.perf_counter() - chunk_started) * 1000, 2),
        )

        if not retrieval_chunks:

            raise RuntimeError(
                "Document processing produced "
                "zero retrieval chunks."
            )

        # ====================================================
        # STEP 4B
        # PostgreSQL
        # ====================================================

        stored_chunks = (
            self.chunk_service
            .persist_chunks(
                document=document,
                chunks=retrieval_chunks,
                replace_existing=True,
            )
        )

        if not stored_chunks:

            raise RuntimeError(
                "Document processing produced "
                "zero PostgreSQL chunks."
            )

        if (
            len(stored_chunks)
            != len(retrieval_chunks)
        ):

            raise RuntimeError(
                "PostgreSQL chunk count does "
                "not match generated chunks. "
                f"Generated={len(retrieval_chunks)}, "
                f"Persisted={len(stored_chunks)}"
            )

        # ====================================================
        # STEP 5
        # Embedding + ChromaDB
        # ====================================================

        indexing_result = (
            self.indexing_pipeline
            .index_document(
                retrieval_chunks,
                document_id=str(
                    document.id
                ),
                user_id=str(
                    owner.id
                ),
                reindex=reindex,
            )
        )

        self._validate_indexing_result(
            indexing_result=(
                indexing_result
            ),
            stage="complete",
        )

        return (
            parsed_document,
            retrieval_chunks,
            stored_chunks,
            indexing_result,
        )

    # ========================================================
    # OWNERSHIP VALIDATION
    # ========================================================

    @staticmethod
    def _validate_owner(
        *,
        document: Document,
        owner: User,
    ) -> None:
        """
        Verify document ownership before any RAG processing.
        """

        if document is None:

            raise ValueError(
                "document is required."
            )

        if owner is None:

            raise ValueError(
                "owner is required."
            )

        if (
            document.owner_id
            != owner.id
        ):

            raise PermissionError(
                "Document does not belong "
                "to the current user."
            )

    # ========================================================
    # INDEX RESULT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_indexing_result(
        *,
        indexing_result: dict[
            str,
            Any,
        ],
        stage: str,
    ) -> None:
        """
        Validate output from the frozen IndexingPipeline.
        """

        if not isinstance(
            indexing_result,
            dict,
        ):

            raise RuntimeError(
                f"{stage} indexing returned "
                "an invalid result."
            )

        if (
            indexing_result.get(
                "status"
            )
            != "success"
        ):

            raise RuntimeError(
                f"{stage} ChromaDB indexing "
                "did not complete successfully. "
                f"Result={indexing_result}"
            )

    # ========================================================
    # DELETE DOCUMENT VECTORS
    # ========================================================

    def delete_document_vectors(
        self,
        document_id: UUID | str,
    ) -> dict[str, Any]:
        """
        Remove the vector representation of a document.

        Relational/file deletion remains the responsibility of
        DocumentService.delete_document().
        """

        return (
            self.indexing_pipeline
            .delete_document(
                str(
                    document_id
                )
            )
        )