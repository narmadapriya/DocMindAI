from uuid import UUID
from datetime import datetime

from pydantic import BaseModel


# ----------------------------------------
# Upload Response
# ----------------------------------------

class UploadResponse(BaseModel):

    id: UUID

    filename: str

    original_filename: str

    file_type: str

    file_size: int

    created_at: datetime

    class Config:
        from_attributes = True


# ----------------------------------------
# Document Response
# ----------------------------------------

class DocumentResponse(BaseModel):

    id: UUID

    filename: str

    original_filename: str

    file_path: str

    file_type: str

    file_size: int

    owner_id: UUID

    created_at: datetime

    updated_at: datetime

    class Config:
        from_attributes = True


# ----------------------------------------
# Document List
# ----------------------------------------

class DocumentListResponse(BaseModel):

    documents: list[DocumentResponse]


# ----------------------------------------
# Delete Response
# ----------------------------------------

class DeleteDocumentResponse(BaseModel):

    success: bool

    message: str

# ----------------------------------------
# Chunk Detail
# ----------------------------------------

class ChunkDetailResponse(BaseModel):

    id: UUID

    chunk_index: int

    content: str

    chunk_type: str

    page_number: int | None = None

    created_at: datetime

    class Config:
        from_attributes = True


# ----------------------------------------
# Per-Document Chunk Response
# ----------------------------------------

class DocumentChunkResponse(BaseModel):

    document_id: UUID

    filename: str

    file_type: str

    chunk_count: int

    chunks: list[ChunkDetailResponse]


# ----------------------------------------
# Per-Document Chunk Summary
# ----------------------------------------

class DocumentChunkSummary(BaseModel):

    document_id: UUID

    filename: str

    file_type: str

    chunk_count: int


# ----------------------------------------
# Total Chunk Response
# ----------------------------------------

class TotalDocumentChunksResponse(BaseModel):

    total_documents: int

    documents_with_chunks: int

    documents_without_chunks: int

    total_chunks: int

    documents: list[DocumentChunkSummary]
