from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db
from app.models.chat import Chat, Message
from app.models.chunk import Chunk
from app.models.citation import Citation
from app.models.document import Document
from app.models.user import User
from app.rag.vectordb.chroma_manager import ChromaManager


router = APIRouter(
    prefix="/citations",
    tags=["Citations"],
)

chunk_router = APIRouter(
    prefix="/chunks",
    tags=["Citations"],
)


def _safe_uuid(value: str, label: str) -> UUID:
    try:
        return UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid {label}.",
        ) from exc


def _vector_metadata(
    chunk: Chunk,
    document: Document,
) -> tuple[str | None, dict[str, Any]]:
    try:
        result = (
            ChromaManager()
            .get_collection()
            .get(
                where={
                    "document_id":
                        str(document.id)
                },
                include=[
                    "metadatas",
                    "documents",
                ],
            )
        )

        ids = (
            result.get("ids", [])
            or []
        )

        metadatas = (
            result.get(
                "metadatas",
                [],
            )
            or []
        )

        vector_documents = (
            result.get(
                "documents",
                [],
            )
            or []
        )

        for index, vector_id in enumerate(
            ids
        ):
            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            vector_text = (
                vector_documents[index]
                if index
                < len(vector_documents)
                else None
            )

            metadata_chunk_index = (
                metadata.get(
                    "chunk_index"
                )
            )

            if (
                metadata_chunk_index
                is not None
            ):
                try:
                    same_index = (
                        int(
                            metadata_chunk_index
                        )
                        ==
                        int(
                            chunk.chunk_index
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    same_index = False
            else:
                same_index = False

            if (
                same_index
                or (
                    vector_text
                    and vector_text
                    == chunk.content
                )
            ):
                cleaned = dict(
                    metadata
                    or {}
                )

                cleaned[
                    "vector_id"
                ] = vector_id

                return (
                    str(vector_id),
                    cleaned,
                )

    except Exception:
        pass

    return None, {}


def _latest_owned_citation(
    db: Session,
    chunk_id: UUID,
    user_id: UUID,
) -> Citation | None:
    return (
        db.query(
            Citation
        )
        .join(
            Message,
            Citation.message_id
            == Message.id,
        )
        .join(
            Chat,
            Message.chat_id
            == Chat.id,
        )
        .filter(
            Citation.chunk_id
            == chunk_id,
            Chat.user_id
            == user_id,
        )
        .order_by(
            Citation.created_at.desc()
        )
        .first()
    )


def _owned_chunk(
    db: Session,
    chunk_id: UUID,
    user_id: UUID,
) -> tuple[Chunk, Document]:
    row = (
        db.query(
            Chunk,
            Document,
        )
        .join(
            Document,
            Chunk.document_id
            == Document.id,
        )
        .filter(
            Chunk.id
            == chunk_id,
            Document.owner_id
            == user_id,
        )
        .first()
    )

    if not row:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Chunk not found.",
        )

    return row[0], row[1]


def _chunk_payload(
    db: Session,
    chunk: Chunk,
    document: Document,
    user_id: UUID,
    citation: Citation | None = None,
) -> dict[str, Any]:
    if citation is None:
        citation = (
            _latest_owned_citation(
                db,
                chunk.id,
                user_id,
            )
        )

    (
        vector_id,
        vector_metadata,
    ) = _vector_metadata(
        chunk,
        document,
    )

    page = (
        citation.page_number
        if (
            citation
            and citation.page_number
            is not None
        )
        else chunk.page_number
    )

    sheet_name = (
        vector_metadata.get(
            "sheet_name"
        )
    )

    source_location = (
        vector_metadata.get(
            "source_location"
        )
    )

    metadata: dict[
        str,
        Any,
    ] = {
        "chunk_index":
            chunk.chunk_index,

        "chunk_type":
            chunk.chunk_type,

        "document_id":
            str(document.id),

        "file_type":
            document.file_type,

        "vector_id":
            vector_id,

        "sheet_name":
            sheet_name,

        "source_location":
            source_location,
    }

    for key, value in (
        vector_metadata.items()
    ):
        if key not in metadata:
            metadata[key] = value

    return {
        "citation_id":
            str(citation.id)
            if citation
            else None,

        "document_id":
            str(document.id),

        "document_name":
            (
                document.original_filename
                or document.filename
            ),

        "filename":
            (
                document.original_filename
                or document.filename
            ),

        "page":
            page,

        "page_number":
            page,

        "sheet_name":
            sheet_name,

        "chunk_id":
            str(chunk.id),

        "vector_id":
            vector_id,

        "chunk_index":
            chunk.chunk_index,

        "chunk_type":
            chunk.chunk_type,

        "chunk_text":
            chunk.content,

        "content":
            chunk.content,

        "quoted_text":
            (
                citation.quoted_text
                if citation
                else None
            ),

        "confidence_score":
            (
                float(
                    citation.relevance_score
                )
                if (
                    citation
                    and citation.relevance_score
                    is not None
                )
                else None
            ),

        "relevance_score":
            (
                float(
                    citation.relevance_score
                )
                if (
                    citation
                    and citation.relevance_score
                    is not None
                )
                else None
            ),

        "source":
            (
                source_location
                or (
                    document.original_filename
                    or document.filename
                )
            ),

        "metadata":
            metadata,

        "rag_pipeline_details": {
            "citation_persisted":
                citation is not None,

            "postgres_chunk":
                True,

            "vector_metadata_found":
                bool(
                    vector_metadata
                ),

            "vector_store":
                (
                    "ChromaDB"
                    if vector_metadata
                    else None
                ),
        },

        "created_at":
            (
                chunk.created_at.isoformat()
                if chunk.created_at
                else None
            ),
    }


# ==========================================================
# Citation Deduplication
# ==========================================================

def _citation_identity(
    citation: Citation,
    chunk: Chunk,
    document: Document,
) -> tuple[
    str,
    int | None,
    str,
]:
    """
    Response-level citation identity.

    Exact deduplication key:

        document_id
        + page_number
        + chunk_id

    This function does NOT alter citation persistence,
    relevance scores, metadata, ranking, content, or RAG.
    """

    page_number = (
        citation.page_number
        if citation.page_number
        is not None
        else chunk.page_number
    )

    return (
        str(document.id),
        page_number,
        str(chunk.id),
    )


def _serialize_citation_list_item(
    citation: Citation,
    chunk: Chunk,
    document: Document,
) -> dict[str, Any]:
    page_number = (
        citation.page_number
        if citation.page_number
        is not None
        else chunk.page_number
    )

    return {
        "id":
            str(citation.id),

        "citation_id":
            str(citation.id),

        "message_id":
            str(citation.message_id),

        "chunk_id":
            str(chunk.id),

        "document_id":
            str(document.id),

        "document_name":
            (
                document.original_filename
                or document.filename
            ),

        "filename":
            (
                document.original_filename
                or document.filename
            ),

        "page":
            page_number,

        "page_number":
            page_number,

        "chunk_type":
            chunk.chunk_type,

        "quoted_text":
            citation.quoted_text,

        "confidence_score":
            citation.relevance_score,

        "relevance_score":
            citation.relevance_score,

        "created_at":
            (
                citation.created_at.isoformat()
                if citation.created_at
                else None
            ),
    }


@router.get("")
@router.get("/")
def list_citations(
    limit: int = Query(
        default=100,
        ge=1,
        le=250,
    ),
    db: Session = Depends(
        get_db
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    rows = (
        db.query(
            Citation,
            Chunk,
            Document,
        )
        .join(
            Message,
            Citation.message_id
            == Message.id,
        )
        .join(
            Chat,
            Message.chat_id
            == Chat.id,
        )
        .join(
            Chunk,
            Citation.chunk_id
            == Chunk.id,
        )
        .join(
            Document,
            Chunk.document_id
            == Document.id,
        )
        .filter(
            Chat.user_id
            == current_user.id,

            Document.owner_id
            == current_user.id,
        )
        .order_by(
            Citation.created_at.desc()
        )
        .limit(
            limit
        )
        .all()
    )

    # Keep the existing newest-first ranking.
    #
    # If the same persisted RAG citation appears multiple
    # times across messages/conversations, only the first
    # response item for the exact composite key is returned.
    #
    # No DB rows are deleted or modified.

    seen: set[
        tuple[
            str,
            int | None,
            str,
        ]
    ] = set()

    citations: list[
        dict[str, Any]
    ] = []

    for (
        citation,
        chunk,
        document,
    ) in rows:

        identity = (
            _citation_identity(
                citation,
                chunk,
                document,
            )
        )

        if identity in seen:
            continue

        seen.add(
            identity
        )

        citations.append(
            _serialize_citation_list_item(
                citation,
                chunk,
                document,
            )
        )

    return {
        "citations":
            citations
    }


@router.get(
    "/chunk/{chunk_id}"
)
def citation_by_chunk(
    chunk_id: str,
    db: Session = Depends(
        get_db
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    parsed = _safe_uuid(
        chunk_id,
        "chunk id",
    )

    chunk, document = (
        _owned_chunk(
            db,
            parsed,
            current_user.id,
        )
    )

    return _chunk_payload(
        db,
        chunk,
        document,
        current_user.id,
    )


@router.get(
    "/{citation_id}"
)
def citation_detail(
    citation_id: str,
    db: Session = Depends(
        get_db
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    parsed = _safe_uuid(
        citation_id,
        "citation id",
    )

    row = (
        db.query(
            Citation,
            Chunk,
            Document,
        )
        .join(
            Message,
            Citation.message_id
            == Message.id,
        )
        .join(
            Chat,
            Message.chat_id
            == Chat.id,
        )
        .join(
            Chunk,
            Citation.chunk_id
            == Chunk.id,
        )
        .join(
            Document,
            Chunk.document_id
            == Document.id,
        )
        .filter(
            Citation.id
            == parsed,

            Chat.user_id
            == current_user.id,

            Document.owner_id
            == current_user.id,
        )
        .first()
    )

    if not row:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=(
                "Citation not found."
            ),
        )

    (
        citation,
        chunk,
        document,
    ) = row

    return _chunk_payload(
        db,
        chunk,
        document,
        current_user.id,
        citation=citation,
    )


@chunk_router.get(
    "/{chunk_id}"
)
def chunk_detail(
    chunk_id: str,
    db: Session = Depends(
        get_db
    ),
    current_user: User = Depends(
        get_current_user
    ),
):
    parsed = _safe_uuid(
        chunk_id,
        "chunk id",
    )

    chunk, document = (
        _owned_chunk(
            db,
            parsed,
            current_user.id,
        )
    )

    return _chunk_payload(
        db,
        chunk,
        document,
        current_user.id,
    )