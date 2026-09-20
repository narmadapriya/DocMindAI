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

from app.schemas.compare import (
    CompareRequest,
    CompareResponse,
    ComparisonRow,
)

from app.services.rag_application_service import (
    RAGApplicationService,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Comparison"],
)


@router.post(
    "/compare",
    response_model=CompareResponse,
)
def compare(
    request: CompareRequest,

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

        result = service.compare(
            user=current_user,
            document_ids=(
                request.document_ids
            ),
            metrics=request.metrics,
            top_k=request.top_k,
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

    rows = [
        ComparisonRow(
            **row
        )
        for row
        in result.get(
            "comparison",
            [],
        )
    ]

    return CompareResponse(
        answer=result[
            "answer"
        ],
        comparison=rows,
        citations=result[
            "citations"
        ],
        document_count=result[
            "document_count"
        ],
    )