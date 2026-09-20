import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Chunk(Base):
    """
    Stores document chunks used by the RAG pipeline.
    """

    __tablename__ = "chunks"

    # --------------------------------------------------
    # Primary Key
    # --------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # --------------------------------------------------
    # Document Reference
    # --------------------------------------------------

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # --------------------------------------------------
    # Chunk Information
    # --------------------------------------------------

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    chunk_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="text",
    )

    # --------------------------------------------------
    # Page Information
    # --------------------------------------------------

    page_number: Mapped[int | None] = mapped_column(
        Integer,
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

    document = relationship(
        "Document",
        back_populates="chunks",
    )
    
    citations = relationship(
        "Citation",
        back_populates="chunk",
        cascade="all, delete-orphan",
    )

    # --------------------------------------------------
    # Representation
    # --------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Chunk("
            f"document_id='{self.document_id}', "
            f"chunk_index={self.chunk_index}"
            f")>"
        )