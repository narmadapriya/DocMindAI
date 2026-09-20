from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# =====================================================
# User Response
# =====================================================

class UserResponse(BaseModel):
    id: UUID
    username: str
    email: EmailStr
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


# =====================================================
# Current User Profile
# =====================================================

class UserProfileResponse(BaseModel):
    id: UUID
    username: str
    email: EmailStr
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }


# =====================================================
# Update Profile
# =====================================================

class UserUpdateRequest(BaseModel):

    username: str = Field(
        min_length=3,
        max_length=30,
    )

    email: EmailStr


# =====================================================
# Change Password
# =====================================================

class PasswordUpdateRequest(BaseModel):

    current_password: str

    new_password: str = Field(
        min_length=8,
        max_length=100,
    )