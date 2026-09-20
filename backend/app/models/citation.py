import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Citation(Base):
    """
    Stores source citations associated with an assistant message.

    Each citation points to the document chunk that was used
    as supporting evidence for the generated response.
    """

    __tablename__ = "citations"

    # --------------------------------------------------
    # Primary Key
    # --------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # --------------------------------------------------
    # Message Reference
    # --------------------------------------------------

    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "messages.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # --------------------------------------------------
    # Chunk Reference
    # --------------------------------------------------

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "chunks.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # --------------------------------------------------
    # Citation Information
    # --------------------------------------------------

    page_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    relevance_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    quoted_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # --------------------------------------------------
    # Timestamp
    # --------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # --------------------------------------------------
    # Relationships
    # --------------------------------------------------

    message = relationship(
        "Message",
        back_populates="citations",
    )

    chunk = relationship(
        "Chunk",
        back_populates="citations",
    )

    # --------------------------------------------------
    # Representation
    # --------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Citation("
            f"id='{self.id}', "
            f"message_id='{self.message_id}', "
            f"chunk_id='{self.chunk_id}'"
            f")>"
        )