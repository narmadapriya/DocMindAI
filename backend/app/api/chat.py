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

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)

from app.services.rag_application_service import (
    RAGApplicationService,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Chat"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,

    current_user: User = Depends(
        get_current_user
    ),

    service: RAGApplicationService = Depends(
        get_rag_application_service
    ),
):

    # ---------------------------------------------------------
    # Existing schema still includes user_id.
    #
    # Step 9 does not trust a body-supplied user identifier.
    # It must match the authenticated JWT user.
    # ---------------------------------------------------------

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

        result = service.chat(
            user=current_user,
            query=request.query,
            document_ids=(
                request.document_ids
            ),
            conversation_id=(
                request.conversation_id
            ),
            history=[
                item.model_dump()
                for item
                in request.history
            ],
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

    return ChatResponse(
        conversation_id=(
            result[
                "conversation_id"
            ]
        ),
        answer=result[
            "answer"
        ],
        citations=result[
            "citations"
        ],
        evidence_count=result[
            "evidence_count"
        ],
        document_count=result[
            "document_count"
        ],
        conversation_context_used=(
            result[
                "conversation_context_used"
            ]
        ),
    )