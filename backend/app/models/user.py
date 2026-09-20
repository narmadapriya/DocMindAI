import uuid
from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class User(Base):
    """
    Stores application users.
    """

    __tablename__ = "users"

    # --------------------------------------------------
    # Primary Key
    # --------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # --------------------------------------------------
    # Username
    # --------------------------------------------------

    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )

    # --------------------------------------------------
    # Email
    # --------------------------------------------------

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    # --------------------------------------------------
    # Password
    # --------------------------------------------------

    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
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

    documents = relationship(
        "Document",
        back_populates="owner",
        cascade="all, delete-orphan",
    )
    
    chats = relationship(
        "Chat",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    
    analytics = relationship(
        "Analytics",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    
    settings = relationship(
    "Settings",
    back_populates="user",
    cascade="all, delete-orphan",
    )

    # --------------------------------------------------
    # Representation
    # --------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<User(username='{self.username}', "
            f"email='{self.email}')>"
        )