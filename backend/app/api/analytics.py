from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Sequence

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db
from app.models.analytics import Analytics
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.user import User
from app.rag.vectordb.chroma_manager import ChromaManager


router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
)


def _utc_day_start(days: int) -> datetime:
    today = datetime.utcnow().date()
    start = today - timedelta(days=days - 1)
    return datetime.combine(start, datetime.min.time())


def _query_events(db: Session, user_id, days: int) -> list[Analytics]:
    start = _utc_day_start(days)
    return (
        db.query(Analytics)
        .filter(
            Analytics.user_id == user_id,
            Analytics.created_at >= start,
            Analytics.query_count > 0,
        )
        .order_by(Analytics.created_at.asc())
        .all()
    )


def _trend(events: list[Analytics], days: int) -> list[dict[str, Any]]:
    today = datetime.utcnow().date()
    counts: Counter = Counter()

    for event in events:
        if not event.created_at:
            continue
        counts[event.created_at.date()] += int(event.query_count or 0)

    points: list[dict[str, Any]] = []
    for offset in range(days - 1, -1, -1):
        day = today - timedelta(days=offset)
        points.append(
            {
                "date": day.isoformat(),
                "day": day.strftime("%a"),
                "queries": int(counts.get(day, 0)),
            }
        )
    return points


def _document_type(filename: str | None, file_type: str | None) -> str:
    value = (file_type or "").strip().replace(".", "")
    if not value and filename and "." in filename:
        value = filename.rsplit(".", 1)[-1]
    return (value or "OTHER").upper()


def _chroma_active_document_count(
    document_ids: Sequence[Any],
) -> tuple[int | None, str]:
    """
    Count only vectors belonging to PostgreSQL documents that
    still exist for the authenticated user.

    Historical/orphan Chroma vectors must not inflate dashboard
    metrics after their document rows have been deleted.
    """

    active_ids = [
        str(document_id)
        for document_id in document_ids
        if document_id is not None
    ]

    if not active_ids:
        return 0, "operational"

    try:
        chroma = ChromaManager()
        count = sum(
            chroma.document_count(document_id)
            for document_id in active_ids
        )
        return int(count), "operational"
    except Exception:
        return None, "unavailable"


@router.get("/queries-over-time")
def queries_over_time(
    days: int = Query(default=7, ge=1, le=90),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    events = _query_events(db, current_user.id, days)
    points = _trend(events, days)
    return {
        "days": days,
        "total_queries": sum(point["queries"] for point in points),
        "points": points,
    }


@router.get("/dashboard")
def dashboard_analytics(
    days: int = Query(default=7, ge=1, le=90),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    documents = (
        db.query(Document)
        .filter(Document.owner_id == current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )

    document_ids = [document.id for document in documents]

    chunk_count = 0
    if document_ids:
        chunk_count = int(
            db.query(func.count(Chunk.id))
            .filter(Chunk.document_id.in_(document_ids))
            .scalar()
            or 0
        )

    events = _query_events(db, current_user.id, days)
    trend = _trend(events, days)

    all_query_rows = (
        db.query(Analytics.query_count)
        .filter(
            Analytics.user_id == current_user.id,
            Analytics.query_count > 0,
        )
        .all()
    )
    total_queries = int(sum(int(row[0] or 0) for row in all_query_rows))
    response_times = [
        int(event.response_time_ms)
        for event in events
        if event.response_time_ms is not None
    ]
    avg_response_time_ms = (
        round(sum(response_times) / len(response_times), 2)
        if response_times
        else None
    )

    type_counts: Counter = Counter(
        _document_type(document.original_filename, document.file_type)
        for document in documents
    )

    storage_used_bytes = int(
        sum(int(document.file_size or 0) for document in documents)
    )

    vector_count, vector_status = _chroma_active_document_count(
        document_ids
    )

    recent_activity: list[dict[str, Any]] = []
    for document in documents[:5]:
        document_chunk_count = 0
        if document.id:
            document_chunk_count = int(
                db.query(func.count(Chunk.id))
                .filter(Chunk.document_id == document.id)
                .scalar()
                or 0
            )

        recent_activity.append(
            {
                "id": str(document.id),
                "filename": document.original_filename or document.filename,
                "file_type": _document_type(
                    document.original_filename,
                    document.file_type,
                ),
                "file_size": int(document.file_size or 0),
                "created_at": (
                    document.created_at.isoformat()
                    if document.created_at
                    else None
                ),
                "chunk_count": document_chunk_count,
                "status": "ready" if document_chunk_count > 0 else "uploaded",
            }
        )

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "period_days": days,
        "metrics": {
            "total_documents": len(documents),
            "chunks_indexed": chunk_count,
            "queries_asked": total_queries,
            "storage_used_bytes": storage_used_bytes,
            "embeddings_generated": vector_count,
            "avg_response_time_ms": avg_response_time_ms,
        },
        "queries_over_time": trend,
        "document_types": [
            {"name": name, "value": int(value)}
            for name, value in sorted(type_counts.items())
        ],
        "recent_activity": recent_activity,
        "system_status": [
            {"name": "Document Processing", "status": "operational"},
            {"name": "PostgreSQL", "status": "operational"},
            {"name": "Vector Database", "status": vector_status},
            {
                "name": "Embedding Index",
                "status": vector_status,
            },
            {"name": "Agentic RAG API", "status": "operational"},
        ],
    }
