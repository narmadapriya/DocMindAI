from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import os
from typing import Any

from jose import JWTError, jwt

from app.core.security import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    ACCESS_TOKEN_TYPE,
    REFRESH_TOKEN_TYPE,
)


PASSWORD_RESET_TOKEN_TYPE = "password_reset"


def _password_version(hashed_password: str) -> str:
    """Return a non-reversible fingerprint used to invalidate reset links."""
    return sha256(hashed_password.encode("utf-8")).hexdigest()


class JWTHandler:
    """
    Handles JWT Access, Refresh, and short-lived Password Reset tokens.
    """

    def create_access_token(self, subject: str) -> str:

        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

        payload = {
            "sub": str(subject),
            "type": ACCESS_TOKEN_TYPE,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }

        return jwt.encode(
            payload,
            SECRET_KEY,
            algorithm=ALGORITHM,
        )

    def create_refresh_token(self, subject: str) -> str:

        expire = datetime.now(timezone.utc) + timedelta(
            days=REFRESH_TOKEN_EXPIRE_DAYS
        )

        payload = {
            "sub": str(subject),
            "type": REFRESH_TOKEN_TYPE,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }

        return jwt.encode(
            payload,
            SECRET_KEY,
            algorithm=ALGORITHM,
        )

    def create_password_reset_token(
        self,
        subject: str,
        hashed_password: str,
    ) -> str:
        """
        Create a short-lived reset token bound to the user's current password.

        The password fingerprint makes the token effectively one-time: after a
        successful password reset the stored hash changes and any old reset
        link becomes invalid immediately, even if its JWT expiry has not elapsed.
        """
        try:
            expire_minutes = int(
                os.getenv(
                    "DOCMINDAI_PASSWORD_RESET_EXPIRE_MINUTES",
                    "15",
                )
            )
        except ValueError:
            expire_minutes = 15

        expire_minutes = max(5, min(expire_minutes, 60))
        now = datetime.now(timezone.utc)

        payload = {
            "sub": str(subject),
            "type": PASSWORD_RESET_TOKEN_TYPE,
            "pwdv": _password_version(hashed_password),
            "exp": now + timedelta(minutes=expire_minutes),
            "iat": now,
        }

        return jwt.encode(
            payload,
            SECRET_KEY,
            algorithm=ALGORITHM,
        )

    def decode_token(
        self,
        token: str,
    ) -> dict[str, Any]:

        return jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )

    def verify_token(
        self,
        token: str,
        token_type: str,
    ) -> dict[str, Any]:

        try:

            payload = self.decode_token(token)

            if payload.get("type") != token_type:
                raise ValueError("Invalid token type.")

            return payload

        except JWTError as exc:
            raise ValueError("Invalid or expired token.") from exc

    def verify_password_reset_token(
        self,
        token: str,
        hashed_password: str,
    ) -> dict[str, Any]:
        """Validate reset-token signature, expiry, type, and password binding."""
        payload = self.verify_token(
            token,
            PASSWORD_RESET_TOKEN_TYPE,
        )

        supplied_version = str(payload.get("pwdv") or "")
        expected_version = _password_version(hashed_password)

        if not supplied_version or not hmac.compare_digest(
            supplied_version,
            expected_version,
        ):
            raise ValueError("Invalid or expired token.")

        return payload


# --------------------------------------------------
# Singleton
# --------------------------------------------------

jwt_handler = JWTHandler()


# --------------------------------------------------
# Wrapper Functions
# --------------------------------------------------

def create_access_token(subject: str) -> str:
    return jwt_handler.create_access_token(subject)


def create_refresh_token(subject: str) -> str:
    return jwt_handler.create_refresh_token(subject)


def create_password_reset_token(
    subject: str,
    hashed_password: str,
) -> str:
    return jwt_handler.create_password_reset_token(
        subject,
        hashed_password,
    )


def decode_token(token: str):
    return jwt_handler.decode_token(token)


def verify_token(
    token: str,
    token_type: str,
):
    return jwt_handler.verify_token(
        token,
        token_type,
    )


def verify_password_reset_token(
    token: str,
    hashed_password: str,
):
    return jwt_handler.verify_password_reset_token(
        token,
        hashed_password,
    )
