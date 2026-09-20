from __future__ import annotations

from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from fastapi.responses import (
    FileResponse,
)

from sqlalchemy import func

from sqlalchemy.orm import Session

from app.auth.dependencies import (
    get_current_user,
)

from app.database.session import (
    get_db,
)

from app.models.chunk import (
    Chunk,
)

from app.models.document import (
    Document,
)

from app.models.user import (
    User,
)

from app.rag.vectordb import (
    ChromaManager,
)

from app.schemas.document import (
    DeleteDocumentResponse,
    DocumentChunkResponse,
    DocumentListResponse,
    TotalDocumentChunksResponse,
    UploadResponse,
)

from app.services.background_ingestion_service import (
    clear_document_processing_state,
    get_document_processing_state,
    mark_document_scheduled,
    process_document_in_background,
)

from app.services.document_service import (
    DocumentService,
)


# ==========================================================
# Router
# ==========================================================

router = APIRouter(
    prefix="/upload",
    tags=["Document Upload"],
)


# ==========================================================
# Shared Read-Only Status Chroma Manager
# ==========================================================

# ChromaManager creates its PersistentClient/collection lazily.
# Reusing this manager avoids reconstructing those objects on every
# frontend status-poll request while preserving the same collection,
# metadata, and document-count behaviour.
_status_chroma = ChromaManager()


# ==========================================================
# Internal Ownership Helper
# ==========================================================

def _get_owned_document(
    *,
    db: Session,
    document_id: UUID,
    current_user: User,
) -> Document:
    """
    Load one document and enforce ownership.

    This helper is used by status/delete operations so another
    user's document can never be inspected or have its vectors
    removed.
    """

    document = (
        db.query(Document)
        .filter(
            Document.id
            == document_id,

            Document.owner_id
            == current_user.id,
        )
        .first()
    )

    if document is None:

        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=(
                "Document not found."
            ),
        )

    return document


# ==========================================================
# Upload Single File
# ==========================================================

@router.post(
    "/",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    background_tasks: BackgroundTasks,

    file: UploadFile = File(...),

    db: Session = Depends(
        get_db
    ),

    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Fast document upload.

    Synchronous request path:

        UploadFile
            ↓
        DocumentService
            ↓
        physical file
            ↓
        PostgreSQL Document
            ↓
        HTTP 201

    Heavy processing runs after the response:

        ParserService
            ↓
        Multimodal Processing
            ↓
        Chunking
            ↓
        PostgreSQL chunks
            ↓
        Embeddings
            ↓
        ChromaDB
    """

    try:

        # ==================================================
        # Fast synchronous upload
        # ==================================================

        service = (
            DocumentService(
                db
            )
        )

        document = (
            service.upload_document(
                file=file,
                owner=current_user,
            )
        )

        # --------------------------------------------------
        # Save primitive IDs before request scope ends.
        # --------------------------------------------------

        document_id = (
            document.id
        )

        owner_id = (
            current_user.id
        )

        # --------------------------------------------------
        # Ensure the Document row is visible to the new
        # SQLAlchemy session created by the background worker.
        #
        # DocumentService may already commit internally;
        # this commit is intentionally safe/idempotent.
        # --------------------------------------------------

        db.commit()

        db.refresh(
            document
        )

        # ==================================================
        # Background RAG processing
        # ==================================================

        mark_document_scheduled(
            document_id=document_id,
            owner_id=owner_id,
        )

        background_tasks.add_task(
            process_document_in_background,
            document_id=document_id,
            owner_id=owner_id,
        )

        print(
            "[UPLOAD] Document saved:"
        )

        print(
            f"[UPLOAD] Document ID: "
            f"{document_id}"
        )

        print(
            f"[UPLOAD] File: "
            f"{document.original_filename}"
        )

        print(
            "[UPLOAD] Background ingestion "
            "scheduled."
        )

        return document

    except HTTPException:

        raise

    except ValueError as exc:

        db.rollback()

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(
                exc
            ),
        ) from exc

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Document upload failed: "
                f"{exc}"
            ),
        ) from exc


# ==========================================================
# Upload Multiple Files
# ==========================================================

@router.post(
    "/multiple",
    response_model=list[
        UploadResponse
    ],
    status_code=status.HTTP_201_CREATED,
)
def upload_multiple_documents(
    background_tasks: BackgroundTasks,

    files: list[
        UploadFile
    ] = File(...),

    db: Session = Depends(
        get_db
    ),

    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Upload multiple documents quickly.

    The files/Document rows are created synchronously.

    Heavy parsing, multimodal processing, chunking,
    embedding, and ChromaDB indexing are scheduled as
    background tasks.

    Maximum:
        10 files/request
    """

    # ======================================================
    # Validation
    # ======================================================

    if not files:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "At least one file "
                "is required."
            ),
        )

    if len(files) > 10:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Maximum 10 files allowed."
            ),
        )

    service = (
        DocumentService(
            db
        )
    )

    documents: list[
        Document
    ] = []

    try:

        # ==================================================
        # Save files + Document rows
        # ==================================================

        for file in files:

            document = (
                service.upload_document(
                    file=file,
                    owner=current_user,
                )
            )

            documents.append(
                document
            )

        # --------------------------------------------------
        # Ensure every Document row is committed before
        # background workers create their own DB sessions.
        # --------------------------------------------------

        db.commit()

        # --------------------------------------------------
        # Refresh while request-scoped session is available.
        # --------------------------------------------------

        for document in documents:

            db.refresh(
                document
            )

        # ==================================================
        # Schedule background ingestion
        # ==================================================

        owner_id = (
            current_user.id
        )

        for document in documents:

            mark_document_scheduled(
                document_id=(
                    document.id
                ),
                owner_id=(
                    owner_id
                ),
            )

            background_tasks.add_task(
                process_document_in_background,
                document_id=(
                    document.id
                ),
                owner_id=(
                    owner_id
                ),
            )

        print(
            "[UPLOAD MULTIPLE] "
            f"Saved {len(documents)} "
            "document(s)."
        )

        print(
            "[UPLOAD MULTIPLE] "
            "Background ingestion scheduled."
        )

        return documents

    except HTTPException:

        raise

    except ValueError as exc:

        db.rollback()

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(
                exc
            ),
        ) from exc

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Multiple document upload "
                f"failed: {exc}"
            ),
        ) from exc


# ==========================================================
# List My Documents
# ==========================================================

@router.get(
    "/documents",
    response_model=(
        DocumentListResponse
    ),
)
def list_documents(
    db: Session = Depends(
        get_db
    ),

    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Return documents belonging to the current user.
    """

    service = (
        DocumentService(
            db
        )
    )

    documents = (
        service.get_documents(
            current_user
        )
    )

    return {
        "documents":
            documents
    }




# ==========================================================
# Get Chunks For One Document
# ==========================================================

@router.get(
    "/chunk/{document_id}",
    response_model=DocumentChunkResponse,
)
def get_document_chunks(
    document_id: UUID,

    db: Session = Depends(
        get_db
    ),

    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Return all persisted PostgreSQL chunks for one owned document.

    Swagger:
        GET /api/upload/chunk/{document_id}

    This endpoint is read-only.

    It does not:
        re-parse the file
        regenerate chunks
        create embeddings
        modify ChromaDB
        modify PostgreSQL

    The chunk_count returned here is the authoritative count of
    rows currently persisted in the PostgreSQL ``chunks`` table
    for this document.
    """

    # ======================================================
    # Ownership
    # ======================================================

    document = (
        _get_owned_document(
            db=db,
            document_id=document_id,
            current_user=current_user,
        )
    )

    # ======================================================
    # Persisted PostgreSQL Chunks
    # ======================================================

    chunks = (
        db.query(Chunk)
        .filter(
            Chunk.document_id
            == document.id
        )
        .order_by(
            Chunk.chunk_index.asc()
        )
        .all()
    )

    return {
        "document_id":
            document.id,

        "filename":
            document.original_filename,

        "file_type":
            document.file_type,

        "chunk_count":
            len(chunks),

        "chunks":
            chunks,
    }


# ==========================================================
# Get Total Chunk Count For My Documents
# ==========================================================

@router.get(
    "/chunks/total",
    response_model=TotalDocumentChunksResponse,
)
def get_total_document_chunks(
    db: Session = Depends(
        get_db
    ),

    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Return the current user's total persisted PostgreSQL chunk count
    together with the chunk count for every owned document.

    Swagger:
        GET /api/upload/chunks/total

    This is the endpoint to compare with the Dashboard
    ``Chunks Indexed`` value.

    ``total_chunks`` is the sum of PostgreSQL chunk rows across
    all documents owned by the authenticated user.
    """

    # ======================================================
    # One grouped PostgreSQL query
    # ======================================================

    rows = (
        db.query(
            Document.id.label(
                "document_id"
            ),

            Document.original_filename.label(
                "filename"
            ),

            Document.file_type.label(
                "file_type"
            ),

            Document.created_at.label(
                "created_at"
            ),

            func.count(
                Chunk.id
            ).label(
                "chunk_count"
            ),
        )
        .outerjoin(
            Chunk,
            Chunk.document_id
            == Document.id,
        )
        .filter(
            Document.owner_id
            == current_user.id
        )
        .group_by(
            Document.id,
            Document.original_filename,
            Document.file_type,
            Document.created_at,
        )
        .order_by(
            Document.created_at.desc()
        )
        .all()
    )

    documents = [
        {
            "document_id":
                row.document_id,

            "filename":
                row.filename,

            "file_type":
                row.file_type,

            "chunk_count":
                int(
                    row.chunk_count
                    or 0
                ),
        }
        for row in rows
    ]

    total_chunks = sum(
        document[
            "chunk_count"
        ]
        for document
        in documents
    )

    documents_with_chunks = sum(
        1
        for document
        in documents
        if document[
            "chunk_count"
        ] > 0
    )

    total_documents = (
        len(
            documents
        )
    )

    return {
        "total_documents":
            total_documents,

        "documents_with_chunks":
            documents_with_chunks,

        "documents_without_chunks":
            (
                total_documents
                - documents_with_chunks
            ),

        "total_chunks":
            total_chunks,

        "documents":
            documents,
    }


# ==========================================================
# Document Processing Status
# ==========================================================

@router.get(
    "/status/{document_id}",
)
def document_processing_status(
    document_id: UUID,

    db: Session = Depends(
        get_db
    ),

    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Determine whether background RAG processing has produced:

        PostgreSQL chunks
        +
        ChromaDB vectors

    Status values:

        processing
        ready

    ready_for_rag becomes True only when both PostgreSQL
    chunks and Chroma vectors exist.
    """

    # ======================================================
    # Ownership
    # ======================================================

    document = (
        _get_owned_document(
            db=db,
            document_id=document_id,
            current_user=current_user,
        )
    )

    # ======================================================
    # PostgreSQL Chunk + Page Count
    # ======================================================

    # One aggregate query replaces the previous COUNT query plus
    # a second query that loaded every page_number row. This endpoint
    # is polled frequently by the frontend, so keeping it lightweight
    # directly improves Processing -> Ready refresh time.
    chunk_count, max_page_number = (
        db.query(
            func.count(
                Chunk.id
            ),
            func.max(
                Chunk.page_number
            ),
        )
        .filter(
            Chunk.document_id
            == document.id
        )
        .one()
    )

    chunk_count = int(
        chunk_count or 0
    )

    page_count = (
        int(
            max_page_number
        )
        if max_page_number
        is not None
        else None
    )

    # ======================================================
    # ChromaDB Vector Count
    # ======================================================

    try:

        vector_count = (
            _status_chroma.document_count(
                str(
                    document.id
                )
            )
        )

    except Exception as exc:

        # --------------------------------------------------
        # A Chroma status read should not make the API itself
        # crash. The document simply remains not ready.
        # --------------------------------------------------

        print(
            "[UPLOAD STATUS] "
            "Unable to read ChromaDB:"
        )

        print(
            f"[UPLOAD STATUS] {exc}"
        )

        vector_count = 0

    # ======================================================
    # Determine State
    # ======================================================

    # ------------------------------------------------------
    # Raw index readiness
    # ------------------------------------------------------
    #
    # Stage 1 of background ingestion writes usable text/table
    # chunks and vectors before the optional multimodal stage
    # finishes.  The previous status logic reported the document
    # as ready immediately at that point.  The frontend therefore
    # allowed Chat & Ask, Summary and Comparison requests to run
    # while OCR / Qwen2.5-VL / enriched re-indexing was still using
    # the same local Ollama and Chroma resources.  New documents
    # consequently had much higher first-request latency than
    # already-ingested documents.
    #
    # Keep the existing two-stage ingestion exactly as-is, but do
    # not expose a newly uploaded/reindexed document as RAG-ready
    # until its background job has completely finished.  Existing
    # documents created before this process started have no
    # in-memory processing state, so their current behaviour remains
    # unchanged.

    index_ready = (
        chunk_count > 0
        and vector_count > 0
    )

    background_state = (
        get_document_processing_state(
            document.id
        )
    )

    background_status = (
        str(
            background_state.get(
                "status",
                "",
            )
        ).strip().lower()
        if background_state
        else ""
    )

    background_failed = (
        background_status == "failed"
    )

    background_complete = (
        background_state is None
        or background_status == "ready"
    )

    # This field is consumed by the existing frontend to decide
    # whether a document may be selected for Chat, Summary and
    # Comparison.  Its response name and API contract are preserved;
    # only the readiness condition is made truthful for newly
    # scheduled background jobs.
    ready_for_rag = (
        index_ready
        and background_complete
        and not background_failed
    )

    if background_failed:
        processing_status = "failed"
        processing_error = (
            background_state.get(
                "error"
            )
            or "Background document processing failed."
        )

    elif ready_for_rag:
        processing_status = "ready"
        processing_error = None

    else:
        processing_status = "processing"
        processing_error = None

    return {
        "document_id":
            str(
                document.id
            ),

        "filename":
            document.original_filename,

        "status":
            processing_status,

        "chunk_count":
            chunk_count,

        "vector_count":
            vector_count,

        "page_count":
            page_count,

        "ready_for_rag":
            ready_for_rag,

        # Additive Phase 14 operational fields. Existing callers can
        # ignore them; the route/path and existing response fields are
        # unchanged.
        "processing_error":
            processing_error,

        "processing_attempt": (
            background_state.get(
                "attempt"
            )
            if background_state
            else None
        ),
    }


# ==========================================================
# Re-index Existing Document
# ==========================================================

@router.post(
    "/reindex/{document_id}",
    status_code=status.HTTP_202_ACCEPTED,
)
def reindex_document(
    document_id: UUID,

    background_tasks: BackgroundTasks,

    db: Session = Depends(
        get_db
    ),

    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Re-run the already-frozen ingestion/indexing pipeline for one owned
    document without creating a new document row or changing the schema.

    The existing background worker already performs replace-existing
    PostgreSQL chunk persistence and Chroma re-indexing, so this endpoint
    only verifies ownership and schedules that existing workflow.
    """

    document = _get_owned_document(
        db=db,
        document_id=document_id,
        current_user=current_user,
    )

    mark_document_scheduled(
        document_id=document.id,
        owner_id=current_user.id,
    )

    background_tasks.add_task(
        process_document_in_background,
        document_id=document.id,
        owner_id=current_user.id,
    )

    return {
        "document_id": str(document.id),
        "filename": document.original_filename,
        "status": "processing",
        "message": "Document re-indexing scheduled.",
    }


# ==========================================================
# Download Document
# ==========================================================

@router.get(
    "/download/{document_id}",
)
def download_document(
    document_id: UUID,

    db: Session = Depends(
        get_db
    ),

    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Download one owned document.
    """

    service = (
        DocumentService(
            db
        )
    )

    try:

        document = (
            service.download_document(
                document_id=(
                    document_id
                ),
                owner=(
                    current_user
                ),
            )
        )

        return FileResponse(
            path=(
                document.file_path
            ),
            filename=(
                document.original_filename
            ),
        )

    except HTTPException:

        raise

    except PermissionError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=str(
                exc
            ),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(
                exc
            ),
        ) from exc


# ==========================================================
# Delete Document
# ==========================================================

@router.delete(
    "/{document_id}",
    response_model=(
        DeleteDocumentResponse
    ),
)
def delete_document(
    document_id: UUID,

    db: Session = Depends(
        get_db
    ),

    current_user: User = Depends(
        get_current_user
    ),
):
    """
    Delete an owned document completely.

    Deletion order:

        Verify ownership
            ↓
        Delete ChromaDB vectors
            ↓
        DocumentService.delete_document()
            ↓
        PostgreSQL Document deletion
            ↓
        PostgreSQL Chunk cascade
            ↓
        physical file deletion

    This prevents stale Chroma vectors from remaining after
    the user removes a document.
    """

    # ======================================================
    # Verify ownership BEFORE touching ChromaDB
    # ======================================================

    document = (
        _get_owned_document(
            db=db,
            document_id=document_id,
            current_user=current_user,
        )
    )

    # ======================================================
    # Delete Chroma vectors
    # ======================================================

    try:

        chroma = (
            ChromaManager()
        )

        vector_count = (
            chroma.document_count(
                str(
                    document.id
                )
            )
        )

        if vector_count > 0:

            delete_result = (
                chroma.delete_by_document(
                    str(
                        document.id
                    )
                )
            )

            remaining_vectors = (
                chroma.document_count(
                    str(
                        document.id
                    )
                )
            )

            if remaining_vectors != 0:

                raise RuntimeError(
                    "ChromaDB still contains "
                    f"{remaining_vectors} vector(s) "
                    f"for deleted document {document.id}."
                )

            print(
                "[DELETE] Removed "
                f"{delete_result.get('deleted', vector_count)} "
                "Chroma vector(s) for "
                f"{document.id}"
            )

        else:

            print(
                "[DELETE] No Chroma "
                "vectors found for "
                f"{document.id}"
            )

    except Exception as exc:

        # --------------------------------------------------
        # Stop deletion rather than knowingly leaving stale
        # vector data behind.
        # --------------------------------------------------

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Unable to remove document "
                "from ChromaDB: "
                f"{exc}"
            ),
        ) from exc

    # ======================================================
    # Existing relational/file deletion
    # ======================================================

    service = (
        DocumentService(
            db
        )
    )

    try:

        result = (
            service.delete_document(
                document_id=(
                    document_id
                ),
                owner=(
                    current_user
                ),
            )
        )

        clear_document_processing_state(
            document_id
        )

        # --------------------------------------------------
        # Legacy/orphan-vector cleanup.
        #
        # The current delete path removes the selected document
        # vectors above. Older development builds may have left
        # historical user-tagged vectors whose PostgreSQL document
        # rows no longer exist. If this was the user's last live
        # document, no vector for that user can be valid anymore,
        # so it is safe to remove any remaining user-scoped orphans.
        # --------------------------------------------------

        remaining_documents = int(
            db.query(
                func.count(
                    Document.id
                )
            )
            .filter(
                Document.owner_id
                == current_user.id
            )
            .scalar()
            or 0
        )

        if remaining_documents == 0:

            cleanup_result = (
                chroma.delete_by_user(
                    str(
                        current_user.id
                    )
                )
            )

            orphan_count = int(
                cleanup_result.get(
                    "deleted",
                    0,
                )
                or 0
            )

            if orphan_count > 0:

                print(
                    "[DELETE] Removed "
                    f"{orphan_count} legacy/orphan "
                    "user vector(s) after last "
                    "document deletion."
                )

        return result

    except HTTPException:

        raise

    except PermissionError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=str(
                exc
            ),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Document deletion failed: "
                f"{exc}"
            ),
        ) from exc
