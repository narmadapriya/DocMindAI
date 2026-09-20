from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db

from app.models.user import User

from app.schemas.user import (
    UserProfileResponse,
    UserUpdateRequest,
    PasswordUpdateRequest,
)

from app.services.user_service import UserService

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


# =====================================================
# Get Current User
# =====================================================

@router.get(
    "/me",
    response_model=UserProfileResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return current_user


# =====================================================
# Update Profile
# =====================================================

@router.put(
    "/profile",
    response_model=UserProfileResponse,
)
def update_profile(
    request: UserUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    service = UserService(db)

    # Check email uniqueness
    existing_email = service.get_by_email(request.email)

    if existing_email and existing_email.id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists.",
        )

    # Check username uniqueness
    existing_username = service.get_by_username(request.username)

    if existing_username and existing_username.id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists.",
        )

    return service.update_profile(
        current_user,
        request.username,
        request.email,
    )


# =====================================================
# Change Password
# =====================================================

@router.put("/password")
def change_password(
    request: PasswordUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    service = UserService(db)

    try:

        service.update_password(
            current_user,
            request.current_password,
            request.new_password,
        )

        return {
            "success": True,
            "message": "Password updated successfully.",
        }

    except ValueError as e:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )