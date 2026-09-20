from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user

from app.models.user import User

router = APIRouter(
    prefix="/settings",
    tags=["Settings"],
)


# =====================================================
# Protected Dashboard
# =====================================================

@router.get("/dashboard")
def dashboard(

    current_user: User = Depends(get_current_user),

):

    return {

        "message": "Welcome to DocMindAI Dashboard",

        "user": {

            "id": str(current_user.id),

            "username": current_user.username,

            "email": current_user.email,

        },

    }