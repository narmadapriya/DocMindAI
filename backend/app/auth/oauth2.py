from fastapi.security import OAuth2PasswordBearer

"""
OAuth2 scheme used by FastAPI.

Clients should send:

Authorization: Bearer <access_token>

The tokenUrl should match the login endpoint that
will be created in Chapter 4.
"""

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login",
    scheme_name="JWT Authentication",
    description="Enter a valid JWT access token.",
    
)