from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SummaryRequest(BaseModel):
    user_id: str = Field(
        ...,
        min_length=1,
    )

    document_ids: List[str] = Field(
        ...,
        min_length=1,
    )

    scope: str = "document"

    section: str | None = None

    top_k: int = Field(
        default=8,
        ge=1,
        le=30,
    )


class SummaryResponse(BaseModel):
    summary: str
    citations: List[str]
    scope: str
    evidence_count: int
    document_count: int