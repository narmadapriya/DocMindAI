from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging import get_logger, log_event
from app.core.performance import SLOW_REQUEST_MS

logger = get_logger(__name__)


def register_middleware(app: FastAPI) -> None:
    # Deployment-only CORS configuration.
    # This preserves all existing API, authentication, RAG, database,
    # and business logic while allowing the GitHub Pages frontend to
    # call the FastAPI backend from a different HTTPS origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://narmadapriya.github.io",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        started = time.perf_counter()
        path = request.url.path

        log_event(
            logger,
            "request",
            request_id=request_id,
            method=request.method,
            path=path,
        )

        if "upload" in path.lower():
            log_event(
                logger,
                "upload",
                request_id=request_id,
                method=request.method,
                path=path,
            )

        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000
            log_event(
                logger,
                "request_error",
                level=logging.ERROR,
                request_id=request_id,
                method=request.method,
                path=path,
                latency_ms=round(elapsed, 2),
            )
            raise

        elapsed = (time.perf_counter() - started) * 1000
        slow = elapsed >= SLOW_REQUEST_MS
        response.headers["X-Request-ID"] = request_id

        common = {
            "request_id": request_id,
            "method": request.method,
            "path": path,
            "status_code": response.status_code,
            "latency_ms": round(elapsed, 2),
            "slow": slow,
        }
        log_event(logger, "latency", **common)
        log_event(logger, "performance", **common)

        if slow:
            log_event(logger, "slow_request", level=logging.WARNING, **common)

        return response
