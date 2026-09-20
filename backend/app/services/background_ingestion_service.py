from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any
from uuid import UUID

from app.core.logging import get_logger, log_event
from app.database.session import SessionLocal
from app.rag.pipelines.indexing_pipeline import IndexingPipeline
from app.services.ingestion_service import DocumentIngestionService

logger = get_logger(__name__)


# ============================================================
# In-process processing state
# ============================================================
#
# No database/schema change is required.  This registry exists only so
# /api/upload/status/{document_id} can distinguish a real background
# failure from a job that is still running.  Without it, any exception in
# parsing/embedding/ChromaDB was logged and swallowed, while the frontend
# displayed "Processing" forever.
#
# The registry is intentionally bounded for an 8-GB local environment.

_MAX_PROCESSING_STATES = 256
_processing_states: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_processing_state_lock = threading.RLock()

# Heavy OCR/Qwen enrichment is intentionally kept off the critical
# Processing -> Ready path.  The delay gives newly-ready documents a
# foreground window for Chat/Summary/Comparison before local vision
# inference starts on an 8-GB machine.
_FAST_READY_MULTIMODAL_DELAY_SECONDS = max(
    0.0,
    float(
        os.getenv(
            "DOCMIND_FAST_READY_MULTIMODAL_DELAY_SECONDS",
            "30",
        )
    ),
)


def _state_key(document_id: UUID | str) -> str:
    return str(document_id)


def _set_processing_state(
    document_id: UUID | str,
    *,
    status: str,
    owner_id: UUID | str | None = None,
    error: str | None = None,
    error_type: str | None = None,
    attempt: int | None = None,
) -> None:
    key = _state_key(document_id)

    state = {
        "document_id": key,
        "status": status,
        "owner_id": str(owner_id) if owner_id is not None else None,
        "error": error,
        "error_type": error_type,
        "attempt": attempt,
        "updated_at": time.time(),
    }

    with _processing_state_lock:
        _processing_states[key] = state
        _processing_states.move_to_end(key)

        while len(_processing_states) > _MAX_PROCESSING_STATES:
            _processing_states.popitem(last=False)


def mark_document_scheduled(
    *,
    document_id: UUID | str,
    owner_id: UUID | str,
) -> None:
    """Mark a document as queued before FastAPI starts its task."""

    _set_processing_state(
        document_id,
        status="processing",
        owner_id=owner_id,
        attempt=0,
    )


def get_document_processing_state(
    document_id: UUID | str,
) -> dict[str, Any] | None:
    """Return a copy of the latest in-process state, if available."""

    key = _state_key(document_id)

    with _processing_state_lock:
        state = _processing_states.get(key)
        return dict(state) if state is not None else None


def clear_document_processing_state(
    document_id: UUID | str,
) -> None:
    """Remove stale in-process state after document deletion."""

    key = _state_key(document_id)

    with _processing_state_lock:
        _processing_states.pop(key, None)


# ============================================================
# Shared lightweight ingestion resources
# ============================================================

@lru_cache(maxsize=1)
def _shared_indexing_pipeline() -> IndexingPipeline:
    """
    Reuse the Ollama embedding client and Chroma manager across
    background document jobs.

    This does not change embedding vectors, collection names, or RAG
    logic. It only avoids recreating local clients for every upload.
    """

    return IndexingPipeline()


def _warm_embedding_model() -> None:
    """Warm the local embedding model without blocking startup."""

    try:
        pipeline = _shared_indexing_pipeline()
        success = pipeline.embedding_model.warmup()

        if not success:
            log_event(
                logger,
                "embedding_warmup_failed",
                level=30,
                error=(
                    "Embedding warm-up returned False. "
                    "Check that Ollama is running and that "
                    "nomic-embed-text is installed."
                ),
            )

    except Exception as exc:
        # Warm-up is an optimization only. The real ingestion call will
        # retry normally if Ollama becomes available later.
        log_event(
            logger,
            "embedding_warmup_failed",
            level=30,
            error_type=type(exc).__name__,
            error=str(exc),
        )


# Start warming as soon as the upload/background module is imported by
# the FastAPI application. A daemon thread keeps application startup
# non-blocking while usually removing Ollama's cold-load cost before the
# user finishes login/navigation and uploads a document.
_embedding_warmup_thread = threading.Thread(
    target=_warm_embedding_model,
    name="docmindai-embedding-warmup",
    daemon=True,
)
_embedding_warmup_thread.start()


# ============================================================
# Retry policy
# ============================================================

_TRANSIENT_ERROR_HINTS = (
    "ollama",
    "embedding",
    "connection",
    "connecterror",
    "connection refused",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "chromadb",
    "database is locked",
)


def _is_transient_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in message for hint in _TRANSIENT_ERROR_HINTS)


# ============================================================
# Background worker
# ============================================================

def process_document_in_background(
    *,
    document_id: UUID,
    owner_id: UUID,
) -> None:
    """
    Background ingestion worker with a fast-ready boundary.

    Stage 1 remains the existing parser -> chunk -> PostgreSQL ->
    embedding -> Chroma path.  As soon as that usable index exists,
    the processing registry is marked ``ready`` so the existing status
    endpoint can expose the document without waiting for OCR/Qwen2.5-VL.

    Stage 2 remains the existing multimodal enrichment path and still
    runs automatically.  It starts after a short foreground window so
    the user's first Chat/Summary/Comparison request does not compete
    with local vision inference.

    No API, schema, database, authentication, or RAG answer contract is
    changed.  Existing callers of DocumentIngestionService that do not
    supply the callback continue to execute the complete pipeline before
    returning.
    """

    db = SessionLocal()
    max_attempts = 2
    fast_ready_signaled = False

    def mark_fast_ready() -> None:
        nonlocal fast_ready_signaled

        _set_processing_state(
            document_id,
            status="ready",
            owner_id=owner_id,
            attempt=current_attempt,
        )

        fast_ready_signaled = True

        log_event(
            logger,
            "background_ingestion_fast_ready",
            document_id=str(document_id),
            user_id=str(owner_id),
            attempt=current_attempt,
        )

        print(
            "[BACKGROUND][FAST] "
            f"Document {document_id} is ready for RAG."
        )

    current_attempt = 0

    try:
        for attempt in range(1, max_attempts + 1):
            current_attempt = attempt

            _set_processing_state(
                document_id,
                status="processing",
                owner_id=owner_id,
                attempt=attempt,
            )

            log_event(
                logger,
                "background_ingestion_start",
                document_id=str(document_id),
                user_id=str(owner_id),
                attempt=attempt,
            )

            try:
                service = DocumentIngestionService(
                    db,
                    indexing_pipeline=_shared_indexing_pipeline(),
                )

                service.process_existing_document(
                    document_id=document_id,
                    owner_id=owner_id,
                    on_fast_ready=mark_fast_ready,
                    post_fast_ready_delay_seconds=(
                        _FAST_READY_MULTIMODAL_DELAY_SECONDS
                    ),
                )

                # The callback normally sets this as soon as Stage 1
                # finishes.  Keep this final assignment for documents
                # with no callback invocation due to an older/custom
                # ingestion implementation.
                _set_processing_state(
                    document_id,
                    status="ready",
                    owner_id=owner_id,
                    attempt=attempt,
                )

                log_event(
                    logger,
                    "background_ingestion_complete",
                    document_id=str(document_id),
                    user_id=str(owner_id),
                    attempt=attempt,
                    fast_ready=fast_ready_signaled,
                )

                return

            except Exception as exc:
                db.rollback()

                # If Stage 1 was already committed and exposed as ready,
                # a later enrichment failure must not take the usable RAG
                # index back to Failed.  This matches the ingestion
                # service's existing rule that visual enrichment is
                # optional and Stage 1 remains valid.
                if fast_ready_signaled:
                    log_event(
                        logger,
                        "background_multimodal_enrichment_error",
                        level=30,
                        document_id=str(document_id),
                        user_id=str(owner_id),
                        attempt=attempt,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )

                    print(
                        "[BACKGROUND][MULTIMODAL][WARNING] "
                        f"Document {document_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    _set_processing_state(
                        document_id,
                        status="ready",
                        owner_id=owner_id,
                        attempt=attempt,
                    )

                    return

                should_retry = (
                    attempt < max_attempts
                    and _is_transient_error(exc)
                )

                log_event(
                    logger,
                    "background_ingestion_error",
                    level=30 if should_retry else 40,
                    document_id=str(document_id),
                    user_id=str(owner_id),
                    attempt=attempt,
                    retrying=should_retry,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

                print(
                    "[BACKGROUND][ERROR] "
                    f"Document {document_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

                if should_retry:
                    time.sleep(1.5)
                    continue

                _set_processing_state(
                    document_id,
                    status="failed",
                    owner_id=owner_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    attempt=attempt,
                )

                return

    finally:
        db.close()

