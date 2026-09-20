from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.core.exceptions import AppError
from app.core.logging import get_logger, log_event
from app.rag.llm.ollama_client import OllamaModelError
from app.rag.ocr.easyocr_engine import OCREngineError
from app.rag.parser.exceptions import (
    DocumentExtractionError,
    DocumentParsingError,
    UnsupportedDocumentTypeError,
)
from app.rag.vectordb.chroma_manager import ChromaDBError

logger = get_logger(__name__)


def _body(code: str, message: str) -> dict[str, Any]:
    return {"success": False, "error": {"code": code, "message": message}}


def _http_code(status_code: int, message: str) -> str:
    lowered = message.lower()

    if status_code == 400:
        if "unsupported file" in lowered or "unsupported document" in lowered:
            return "UNSUPPORTED_FORMAT"
        if any(token in lowered for token in ("file", "filename", "upload")):
            return "INVALID_FILE"
        return "BAD_REQUEST"

    if status_code == 401:
        return "INVALID_JWT"

    if status_code == 403:
        if "document" in lowered:
            return "UNAUTHORIZED_DOCUMENT_ACCESS"
        return "FORBIDDEN"

    return {
        404: "NOT_FOUND",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
    }.get(status_code, "HTTP_ERROR")


def _ollama_error(exc: OllamaModelError) -> tuple[str, str, int]:
    message = str(exc)
    lowered = message.lower()
    cause = exc.__cause__

    if isinstance(cause, TimeoutError) or "timed out" in lowered or "timeout" in lowered:
        return "MODEL_TIMEOUT", "Model request timed out.", 504

    if (
        "unable to connect" in lowered
        or "connection refused" in lowered
        or "unavailable" in lowered
        or "network/os error" in lowered
    ):
        return "OLLAMA_UNAVAILABLE", "Ollama is unavailable.", 503

    return "LLM_ERROR", "Local model request failed.", 502


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        log_event(
            logger,
            "application_error",
            level=logging.WARNING,
            code=exc.code,
            path=request.url.path,
        )
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(exc.code, exc.message),
            headers=headers,
        )

    @app.exception_handler(UnsupportedDocumentTypeError)
    async def unsupported_document_handler(request: Request, exc: UnsupportedDocumentTypeError):
        log_event(logger, "parsing_error", level=logging.WARNING, code="UNSUPPORTED_FORMAT", path=request.url.path)
        return JSONResponse(status_code=400, content=_body("UNSUPPORTED_FORMAT", "Unsupported document format."))

    @app.exception_handler(DocumentExtractionError)
    async def document_extraction_handler(request: Request, exc: DocumentExtractionError):
        log_event(logger, "parsing_error", level=logging.ERROR, code="DOCUMENT_PARSE_ERROR", path=request.url.path, error=str(exc))
        return JSONResponse(status_code=422, content=_body("DOCUMENT_PARSE_ERROR", "Unable to parse the document"))

    @app.exception_handler(DocumentParsingError)
    async def document_parsing_handler(request: Request, exc: DocumentParsingError):
        log_event(logger, "parsing_error", level=logging.ERROR, code="DOCUMENT_PARSE_ERROR", path=request.url.path, error=str(exc))
        return JSONResponse(status_code=422, content=_body("DOCUMENT_PARSE_ERROR", "Unable to parse the document"))

    @app.exception_handler(OCREngineError)
    async def ocr_handler(request: Request, exc: OCREngineError):
        log_event(logger, "ocr_error", level=logging.ERROR, path=request.url.path, error=str(exc))
        return JSONResponse(status_code=422, content=_body("OCR_ERROR", "OCR processing failed."))

    @app.exception_handler(OllamaModelError)
    async def ollama_handler(request: Request, exc: OllamaModelError):
        code, message, status_code = _ollama_error(exc)
        log_event(logger, "llm_error", level=logging.ERROR, code=code, path=request.url.path, error=str(exc))
        return JSONResponse(status_code=status_code, content=_body(code, message))

    @app.exception_handler(ChromaDBError)
    async def chroma_handler(request: Request, exc: ChromaDBError):
        log_event(logger, "chromadb_error", level=logging.ERROR, path=request.url.path, error=str(exc))
        return JSONResponse(status_code=503, content=_body("CHROMADB_ERROR", "Vector database operation failed."))

    @app.exception_handler(PermissionError)
    async def permission_handler(request: Request, exc: PermissionError):
        message = str(exc)
        code = "UNAUTHORIZED_DOCUMENT_ACCESS" if "document" in message.lower() else "FORBIDDEN"
        public_message = "You are not authorized to access this document." if code == "UNAUTHORIZED_DOCUMENT_ACCESS" else "Access denied."
        log_event(logger, "authorization_error", level=logging.WARNING, code=code, path=request.url.path)
        return JSONResponse(status_code=403, content=_body(code, public_message))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        log_event(logger, "validation_error", level=logging.WARNING, path=request.url.path)
        return JSONResponse(status_code=422, content=_body("VALIDATION_ERROR", "Request validation failed."))

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException):
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        code = _http_code(exc.status_code, message)
        log_event(logger, "http_error", level=logging.WARNING, code=code, path=request.url.path, status_code=exc.status_code)
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(code, message),
            headers=exc.headers,
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_handler(request: Request, exc: SQLAlchemyError):
        log_event(logger, "database_error", level=logging.ERROR, path=request.url.path, error=str(exc))
        return JSONResponse(status_code=503, content=_body("DATABASE_ERROR", "Database operation failed."))

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        log_event(logger, "unhandled_error", level=logging.ERROR, path=request.url.path, error=str(exc))
        return JSONResponse(status_code=500, content=_body("INTERNAL_ERROR", "An unexpected error occurred."))
