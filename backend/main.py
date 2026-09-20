from contextlib import asynccontextmanager
from pathlib import Path
import shutil

from fastapi import FastAPI, Depends
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import configure_logging, get_logger, log_event
from app.core.error_handlers import register_exception_handlers
from app.core.middleware import register_middleware
from app.database.connection import engine

configure_logging()
logger = get_logger(__name__)

from app.models.user import User

from app.auth.dependencies import get_current_user
from app.auth.oauth2 import oauth2_scheme

# ==========================================================
# Routers
# ==========================================================

from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.settings import router as settings_router
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router
from app.api.summary import router as summary_router
from app.api.compare import router as compare_router
from app.api.analytics import router as analytics_router
from app.api.citations import router as citations_router, chunk_router

# ==========================================================
# Clear Python Cache
# ==========================================================

def clear_python_cache():
    """
    Remove all __pycache__ folders recursively.
    """

    project_root = Path(__file__).parent

    cache_count = 0

    for cache in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(cache)
            cache_count += 1
        except Exception:
            pass

    print(f"Cleared {cache_count} Python cache folders")


# ==========================================================
# Startup / Shutdown
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n==============================")
    print("Starting DocMindAI Backend")
    print("==============================")

    # Clear cache
    clear_python_cache()

    # Database Test
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        log_event(logger, "database_connected")
        print("PostgreSQL Connected Successfully")

    except Exception as e:
        log_event(logger, "database_error", error=str(e))
        print(f"Database Connection Failed: {e}")

    print("Backend Ready\n")

    yield

    print("\nDocMindAI Backend Stopped")


# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

register_exception_handlers(app)
register_middleware(app)


# ==========================================================
# Root
# ==========================================================

@app.get("/", tags=["Home"])
def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "status": "Running",
    }


# ==========================================================
# Health
# ==========================================================

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy"
    }


# ==========================================================
# Database Test
# ==========================================================

@app.get("/db-test", tags=["Database"])
def db_test():

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return {
            "database": "connected"
        }

    except Exception as e:
        return {
            "database": "failed",
            "error": str(e)
        }


# ==========================================================
# OAuth2 Token Test
# ==========================================================

@app.get("/token-test", tags=["Authentication"])
def token_test(
    token: str = Depends(oauth2_scheme)
):
    return {
        "received_token": token
    }


# ==========================================================
# Protected Test
# ==========================================================

@app.get("/me", tags=["Authentication"])
def read_me(
    current_user: User = Depends(get_current_user),
):
    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
    }


# ==========================================================
# API Routers
# ==========================================================

app.include_router(
    auth_router,
    prefix="/api",
)

app.include_router(
    users_router,
    prefix="/api",
)

app.include_router(
    settings_router,
    prefix="/api",
)

app.include_router(
    upload_router,
    prefix="/api",
)

app.include_router(chat_router)
app.include_router(summary_router)
app.include_router(compare_router)

# Phase 12 read-only dashboard/citation routes
app.include_router(analytics_router, prefix="/api")
app.include_router(citations_router, prefix="/api")
app.include_router(chunk_router, prefix="/api")

