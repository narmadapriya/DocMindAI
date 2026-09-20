from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
    )

    user_id: str = Field(
        ...,
        min_length=1,
    )

    document_ids: List[str] = Field(
        default_factory=list
    )

    conversation_id: Optional[str] = None

    history: List[ChatMessage] = Field(
        default_factory=list
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: List[str]
    evidence_count: int
    document_count: int
    conversation_context_used: bool