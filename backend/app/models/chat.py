import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class Chat(Base):
    """
    Stores a user's chat/conversation session.
    """

    __tablename__ = "chats"

    # --------------------------------------------------
    # Primary Key
    # --------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # --------------------------------------------------
    # User Reference
    # --------------------------------------------------

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # --------------------------------------------------
    # Chat Information
    # --------------------------------------------------

    title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # --------------------------------------------------
    # Timestamps
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
    # Relationships
    # --------------------------------------------------

    user = relationship(
        "User",
        back_populates="chats",
    )

    messages = relationship(
        "Message",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    # --------------------------------------------------
    # Representation
    # --------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Chat("
            f"id='{self.id}', "
            f"user_id='{self.user_id}', "
            f"title='{self.title}'"
            f")>"
        )


class Message(Base):
    """
    Stores individual messages inside a chat.
    """

    __tablename__ = "messages"

    # --------------------------------------------------
    # Primary Key
    # --------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # --------------------------------------------------
    # Chat Reference
    # --------------------------------------------------

    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "chats.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # --------------------------------------------------
    # Message Information
    # --------------------------------------------------

    role: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
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

    chat = relationship(
        "Chat",
        back_populates="messages",
    )

    citations = relationship(
        "Citation",
        back_populates="message",
        cascade="all, delete-orphan",
    )

    # --------------------------------------------------
    # Representation
    # --------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Message("
            f"id='{self.id}', "
            f"chat_id='{self.chat_id}', "
            f"role='{self.role}'"
            f")>"
        )