from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.jwt_handler import verify_token
from app.auth.oauth2 import oauth2_scheme
from app.core.security import ACCESS_TOKEN_TYPE
from app.core.exceptions import InvalidTokenError
from app.database.session import get_db
from app.models.user import User


credentials_exception = InvalidTokenError("Could not validate credentials")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):

    try:
        payload = verify_token(
            token,
            ACCESS_TOKEN_TYPE,
        )

        user_id = payload.get("sub")

        if user_id is None:
            raise credentials_exception

        user = (
            db.query(User)
            .filter(User.id == UUID(user_id))
            .first()
        )

        if user is None:
            raise credentials_exception

        return user

    except Exception:
        raise credentials_exception