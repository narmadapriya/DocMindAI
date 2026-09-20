from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db

from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    PasswordResetRequest,
    PasswordResetConfirmRequest,
)

from app.schemas.token import (
    TokenResponse,
    RefreshTokenRequest,
)

from fastapi.security import OAuth2PasswordRequestForm

from app.schemas.user import UserResponse

from app.services.auth_service import AuthService

from app.auth.jwt_handler import verify_token

from app.core.security import REFRESH_TOKEN_TYPE

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

# --------------------------------------------------
# Register
# --------------------------------------------------

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=201,
)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):

    service = AuthService(db)

    return service.register(request)


# --------------------------------------------------
# Login
# --------------------------------------------------

@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    service = AuthService(db)

    request = LoginRequest(
        email=form_data.username,
        password=form_data.password,
    )

    return service.login(request)


# --------------------------------------------------
# Password Recovery Request
# --------------------------------------------------

@router.post("/password-reset/request")
def request_password_reset(
    request: PasswordResetRequest,
    db: Session = Depends(get_db),
):
    service = AuthService(db)

    return service.request_password_reset(
        request.email,
    )


# --------------------------------------------------
# Password Recovery Confirm
# --------------------------------------------------

@router.post("/password-reset/confirm")
def confirm_password_reset(
    request: PasswordResetConfirmRequest,
    db: Session = Depends(get_db),
):
    service = AuthService(db)

    return service.reset_password(
        request.token,
        request.new_password,
    )


# --------------------------------------------------
# Refresh Token
# --------------------------------------------------

@router.post(
    "/refresh",
    response_model=TokenResponse,
)
def refresh_token(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db),
):

    payload = verify_token(
        request.refresh_token,
        REFRESH_TOKEN_TYPE,
    )

    user_id = UUID(payload["sub"])

    service = AuthService(db)

    return service.refresh_token(user_id)


# --------------------------------------------------
# Logout
# --------------------------------------------------

@router.post("/logout")
def logout():

    return {
        "success": True,
        "message": "Logged out successfully."
    }
