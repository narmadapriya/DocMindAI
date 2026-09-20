import re

from pydantic import BaseModel, EmailStr, Field, field_validator


# --------------------------------------------------
# Register
# --------------------------------------------------

class RegisterRequest(BaseModel):

    username: str = Field(
        min_length=3,
        max_length=30,
    )

    email: EmailStr

    password: str = Field(
        min_length=8,
        max_length=100,
    )


# --------------------------------------------------
# Login
# --------------------------------------------------

class LoginRequest(BaseModel):

    email: EmailStr

    password: str


# --------------------------------------------------
# Password Recovery
# --------------------------------------------------

class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirmRequest(BaseModel):
    token: str = Field(
        min_length=20,
        max_length=4096,
    )

    new_password: str = Field(
        min_length=8,
        max_length=100,
    )

    @field_validator("new_password")
    @classmethod
    def validate_password_policy(cls, value: str) -> str:
        """Apply the UI's existing 8-4 password rule to recovered passwords."""
        valid = (
            len(value) >= 8
            and bool(re.search(r"[A-Z]", value))
            and bool(re.search(r"[a-z]", value))
            and bool(re.search(r"\d", value))
            and bool(re.search(r"[^A-Za-z0-9]", value))
        )

        if not valid:
            raise ValueError(
                "Password must include at least 8 characters, uppercase and "
                "lowercase letters, a number, and a special character."
            )

        return value


# --------------------------------------------------
# Login Response
# --------------------------------------------------

class LoginResponse(BaseModel):

    access_token: str

    refresh_token: str

    token_type: str = "bearer"
