from uuid import UUID

from sqlalchemy.orm import Session

from fastapi import HTTPException, status

from app.auth.hashing import hash_password, verify_password
from app.auth.jwt_handler import (
    PASSWORD_RESET_TOKEN_TYPE,
    create_access_token,
    create_refresh_token,
    create_password_reset_token,
    verify_password_reset_token,
    verify_token,
)

from app.models.user import User

from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
)

from app.schemas.token import TokenResponse

from app.services.email_service import (
    PasswordResetEmailConfigurationError,
    password_reset_email_configuration_error,
    send_password_reset_email,
)
from app.services.user_service import UserService


class AuthService:
    """
    Handles authentication business logic.

    Features
    --------
    • Register User
    • Login User
    • Refresh Access Token
    • Get Current User
    • Secure Password Recovery
    """

    def __init__(self, db: Session):
        self.db = db
        self.user_service = UserService(db)

    # ----------------------------------------------------
    # Register
    # ----------------------------------------------------
    def register(
        self,
        user: RegisterRequest,
    ) -> User:

        if self.user_service.email_exists(user.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered.",
            )

        if self.user_service.username_exists(user.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists.",
            )

        new_user = self.user_service.create_user(
            username=user.username,
            email=user.email,
            password=user.password,
        )

        return new_user

    # ----------------------------------------------------
    # Login
    # ----------------------------------------------------
    def login(
        self,
        request: LoginRequest,
    ) -> TokenResponse:

        user = self.user_service.get_by_email(
            request.email
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        if not verify_password(
            request.password,
            user.hashed_password,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        access_token = create_access_token(
            user.id
        )

        refresh_token = create_refresh_token(
            user.id
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
        )

    # ----------------------------------------------------
    # Refresh Token
    # ----------------------------------------------------
    def refresh_token(
        self,
        user_id: UUID,
    ) -> TokenResponse:

        user = self.user_service.get_by_id(user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found.",
            )

        access_token = create_access_token(
            user.id
        )

        refresh_token = create_refresh_token(
            user.id
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
        )

    # ----------------------------------------------------
    # Password Reset Request
    # ----------------------------------------------------
    def request_password_reset(
        self,
        email: str,
    ) -> dict[str, object]:
        """
        Email a short-lived password-reset link.

        The public response is intentionally identical whether or not the email
        exists, preventing the endpoint from disclosing registered accounts.
        """
        configuration_error = password_reset_email_configuration_error()
        if configuration_error:
            # This check happens before account lookup so the response cannot be
            # used to infer whether the submitted email is registered.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=configuration_error,
            )

        user = self.user_service.get_by_email(
            email.strip().lower()
        )

        if user:
            token = create_password_reset_token(
                user.id,
                user.hashed_password,
            )

            try:
                send_password_reset_email(
                    user.email,
                    token,
                )
            except PasswordResetEmailConfigurationError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "Password recovery email could not be sent. "
                        "Please try again later."
                    ),
                ) from exc

        return {
            "success": True,
            "message": (
                "If an account exists for that email, a password reset link "
                "has been sent."
            ),
        }

    # ----------------------------------------------------
    # Password Reset Confirm
    # ----------------------------------------------------
    def reset_password(
        self,
        token: str,
        new_password: str,
    ) -> dict[str, object]:
        """Validate a signed, password-bound reset token and replace the hash."""
        invalid_detail = "Reset link is invalid or expired."

        try:
            payload = verify_token(
                token,
                PASSWORD_RESET_TOKEN_TYPE,
            )
            user_id = UUID(str(payload.get("sub")))
        except (TypeError, ValueError, AttributeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=invalid_detail,
            ) from exc

        user = self.user_service.get_by_id(user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=invalid_detail,
            )

        try:
            verify_password_reset_token(
                token,
                user.hashed_password,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=invalid_detail,
            ) from exc

        user.hashed_password = hash_password(new_password)
        self.db.commit()
        self.db.refresh(user)

        return {
            "success": True,
            "message": "Password reset successfully.",
        }

    # ----------------------------------------------------
    # Get User
    # ----------------------------------------------------
    def get_user(
        self,
        user_id: UUID,
    ) -> User:

        user = self.user_service.get_by_id(user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found.",
            )

        return user
