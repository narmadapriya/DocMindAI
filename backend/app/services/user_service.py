from uuid import UUID

from sqlalchemy.orm import Session

from app.models.user import User
from app.auth.hashing import (
    hash_password,
    verify_password,
)


class UserService:
    """
    Handles all user-related database operations.
    """

    def __init__(self, db: Session):
        self.db = db

    # --------------------------------------------------
    # Get User by Email
    # --------------------------------------------------
    def get_by_email(
        self,
        email: str,
    ) -> User | None:

        return (
            self.db.query(User)
            .filter(User.email == email)
            .first()
        )

    # --------------------------------------------------
    # Get User by Username
    # --------------------------------------------------
    def get_by_username(
        self,
        username: str,
    ) -> User | None:

        return (
            self.db.query(User)
            .filter(User.username == username)
            .first()
        )

    # --------------------------------------------------
    # Get User by ID
    # --------------------------------------------------
    def get_by_id(
        self,
        user_id: UUID,
    ) -> User | None:

        return (
            self.db.query(User)
            .filter(User.id == user_id)
            .first()
        )

    # --------------------------------------------------
    # Email Exists
    # --------------------------------------------------
    def email_exists(
        self,
        email: str,
    ) -> bool:

        return self.get_by_email(email) is not None

    # --------------------------------------------------
    # Username Exists
    # --------------------------------------------------
    def username_exists(
        self,
        username: str,
    ) -> bool:

        return self.get_by_username(username) is not None

    # --------------------------------------------------
    # Create User
    # --------------------------------------------------
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
    ) -> User:

        if self.email_exists(email):
            raise ValueError("Email already registered.")

        if self.username_exists(username):
            raise ValueError("Username already exists.")

        user = User(
            username=username,
            email=email,
            hashed_password=hash_password(password),
        )

        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        return user

    # --------------------------------------------------
    # Update Profile
    # --------------------------------------------------
    def update_profile(
        self,
        user: User,
        username: str,
        email: str,
    ) -> User:

        user.username = username
        user.email = email

        self.db.commit()
        self.db.refresh(user)

        return user

    # --------------------------------------------------
    # Update Password
    # --------------------------------------------------
    def update_password(
        self,
        user: User,
        current_password: str,
        new_password: str,
    ) -> User:

        if not verify_password(
            current_password,
            user.hashed_password,
        ):
            raise ValueError("Current password is incorrect.")

        user.hashed_password = hash_password(
            new_password
        )

        self.db.commit()
        self.db.refresh(user)

        return user

    # --------------------------------------------------
    # Delete User
    # --------------------------------------------------
    def delete_user(
        self,
        user: User,
    ):

        self.db.delete(user)
        self.db.commit()