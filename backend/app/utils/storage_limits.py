from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.document import Document


MAX_TOTAL_STORAGE_MB = 500
MAX_TOTAL_STORAGE_BYTES = MAX_TOTAL_STORAGE_MB * 1024 * 1024


def get_user_storage_used_bytes(
    db: Session,
    owner_id: Any,
) -> int:
    """Return persisted document bytes for one authenticated user."""

    total = (
        db.query(func.sum(Document.file_size))
        .filter(Document.owner_id == owner_id)
        .scalar()
    )

    return int(total or 0)


def validate_user_storage_quota(
    db: Session,
    owner_id: Any,
    incoming_size: int,
) -> int:
    """
    Enforce DocMindAI's 500 MB per-user document-storage quota.

    Returns the projected storage usage when the upload is allowed.
    """

    used_bytes = get_user_storage_used_bytes(db, owner_id)
    projected_bytes = used_bytes + int(incoming_size or 0)

    if projected_bytes > MAX_TOTAL_STORAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Total storage limit exceeded. "
                f"Maximum {MAX_TOTAL_STORAGE_MB} MB per user."
            ),
        )

    return projected_bytes
