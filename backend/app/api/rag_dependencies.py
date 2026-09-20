from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from app.database.session import (
    get_db,
)

from app.main_services import (
    get_retrieval_pipeline,
)

from app.services.rag_application_service import (
    RAGApplicationService,
)

from app.services.retrieval_service import (
    RetrievalService,
)


def get_rag_application_service(
    db: Session = Depends(
        get_db
    ),
) -> RAGApplicationService:
    """
    Build the request-scoped application service while reusing only the
    process-wide, DB-independent retrieval pipeline.

    PostgreSQL ownership/evidence verification remains request scoped
    through ``RetrievalService(db=...)``.  The shared pipeline keeps the
    Ollama embedding client and Chroma client/collection warm across
    Chat & Ask requests without changing any API or security boundary.
    """

    retrieval_service = RetrievalService(
        db,
        retrieval_pipeline=(
            get_retrieval_pipeline()
        ),
    )

    return RAGApplicationService(
        db,
        retrieval_service=(
            retrieval_service
        ),
    )
