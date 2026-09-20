import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Settings(Base):
    """
    Stores user-specific application settings.
    """

    __tablename__ = "settings"

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
        unique=True,
        index=True,
    )

    # --------------------------------------------------
    # AI / RAG Settings
    # --------------------------------------------------

    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="qwen2.5:3b",
    )

    embedding_model: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        default="BAAI/bge-small-en-v1.5",
    )

    temperature: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # --------------------------------------------------
    # Application Preferences
    # --------------------------------------------------

    dark_mode: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    notifications_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    # --------------------------------------------------
    # Response Preferences
    # --------------------------------------------------

    citations_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    # --------------------------------------------------
    # Timestamp
    # --------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # --------------------------------------------------
    # Relationship
    # --------------------------------------------------

    user = relationship(
        "User",
        back_populates="settings",
    )

    def __repr__(self):
        return (
            f"<Settings("
            f"user_id='{self.user_id}', "
            f"model_name='{self.model_name}'"
            f")>"
        )