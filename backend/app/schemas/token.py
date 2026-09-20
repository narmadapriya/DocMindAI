from pydantic import BaseModel


class TokenResponse(BaseModel):
    """
    Response returned after successful authentication.
    """

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    """
    Request body for refresh endpoint.
    """

    refresh_token: str


class TokenPayload(BaseModel):
    """
    JWT Payload.
    """

    sub: str
    exp: int