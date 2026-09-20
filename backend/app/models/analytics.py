import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Analytics(Base):
    """
    Stores usage and activity analytics for users.
    """

    __tablename__ = "analytics"

    # --------------------------------------------------
    # Primary Key
    # --------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # --------------------------------------------------
    # User
    # --------------------------------------------------

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # --------------------------------------------------
    # Event Information
    # --------------------------------------------------

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    # --------------------------------------------------
    # Optional Reference
    # --------------------------------------------------

    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # --------------------------------------------------
    # Usage Metrics
    # --------------------------------------------------

    query_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    response_time_ms: Mapped[int | None] = mapped_column(
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
        index=True,
    )

    # --------------------------------------------------
    # Relationships
    # --------------------------------------------------

    user = relationship(
        "User",
        back_populates="analytics",
    )

    document = relationship(
        "Document",
        back_populates="analytics",
    )

    def __repr__(self):
        return (
            f"<Analytics("
            f"user_id='{self.user_id}', "
            f"event_type='{self.event_type}'"
            f")>"
        )