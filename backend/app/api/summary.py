from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.api.rag_dependencies import (
    get_rag_application_service,
)

from app.auth.dependencies import (
    get_current_user,
)

from app.models.user import (
    User,
)

from app.schemas.summary import (
    SummaryRequest,
    SummaryResponse,
)

from app.services.rag_application_service import (
    RAGApplicationService,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Summary"],
)


@router.post(
    "/summary",
    response_model=SummaryResponse,
)
def summary(
    request: SummaryRequest,

    current_user: User = Depends(
        get_current_user
    ),

    service: RAGApplicationService = Depends(
        get_rag_application_service
    ),
):

    if (
        str(
            current_user.id
        )
        != str(
            request.user_id
        )
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=(
                "Request user_id does not "
                "match authenticated user."
            ),
        )

    try:

        result = (
            service.summarize(
                user=current_user,
                document_ids=(
                    request.document_ids
                ),
                scope=request.scope,
                section=request.section,
                top_k=request.top_k,
            )
        )

    except PermissionError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=str(exc),
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(exc),
        ) from exc

    return SummaryResponse(
        summary=result[
            "summary"
        ],
        citations=result[
            "citations"
        ],
        scope=result[
            "scope"
        ],
        evidence_count=result[
            "evidence_count"
        ],
        document_count=result[
            "document_count"
        ],
    )