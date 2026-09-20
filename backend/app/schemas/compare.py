from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    user_id: str = Field(
        ...,
        min_length=1,
    )

    document_ids: List[str] = Field(
        ...,
        min_length=2,
    )

    metrics: Optional[List[str]] = None

    top_k: int = Field(
        default=10,
        ge=1,
        le=40,
    )


class ComparisonRow(BaseModel):
    metric: str
    document_a: str
    document_b: str
    change: str


class CompareResponse(BaseModel):
    answer: str
    comparison: List[ComparisonRow]
    citations: List[str]
    document_count: int