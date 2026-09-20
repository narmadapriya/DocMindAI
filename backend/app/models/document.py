from uuid import uuid4

from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    ForeignKey,
)

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.base import Base


class Document(Base):
    """
    Stores uploaded document metadata.
    """

    __tablename__ = "documents"

    # --------------------------------------------------
    # Primary Key
    # --------------------------------------------------

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    # --------------------------------------------------
    # File Information
    # --------------------------------------------------

    filename = Column(
        String,
        nullable=False,
    )

    original_filename = Column(
        String,
        nullable=False,
    )

    file_path = Column(
        String,
        nullable=False,
    )

    file_type = Column(
        String,
        nullable=False,
    )

    file_size = Column(
        Integer,
        nullable=False,
    )

    # --------------------------------------------------
    # Owner
    # --------------------------------------------------

    owner_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    # --------------------------------------------------
    # Timestamps
    # --------------------------------------------------

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # --------------------------------------------------
    # Relationships
    # --------------------------------------------------

    owner = relationship(
        "User",
        back_populates="documents",
    )

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    
    analytics = relationship(
        "Analytics",
        back_populates="document",
    )

    # --------------------------------------------------
    # Representation
    # --------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Document("
            f"filename='{self.filename}', "
            f"file_type='{self.file_type}', "
            f"owner_id='{self.owner_id}'"
            f")>"
        )