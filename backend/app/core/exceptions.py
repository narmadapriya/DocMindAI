from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AppError(Exception):
    code: str
    message: str
    status_code: int = 400

    def __str__(self) -> str:
        return self.message


class InvalidFileError(AppError):
    def __init__(self, message: str = "Invalid file."):
        super().__init__("INVALID_FILE", message, 400)


class UnsupportedFormatError(AppError):
    def __init__(self, message: str = "Unsupported document format."):
        super().__init__("UNSUPPORTED_FORMAT", message, 400)


class DocumentParseError(AppError):
    def __init__(self, message: str = "Unable to parse the document"):
        super().__init__("DOCUMENT_PARSE_ERROR", message, 422)


class OCRError(AppError):
    def __init__(self, message: str = "OCR processing failed."):
        super().__init__("OCR_ERROR", message, 422)


class OllamaUnavailableError(AppError):
    def __init__(self, message: str = "Ollama is unavailable."):
        super().__init__("OLLAMA_UNAVAILABLE", message, 503)


class ModelTimeoutError(AppError):
    def __init__(self, message: str = "Model request timed out."):
        super().__init__("MODEL_TIMEOUT", message, 504)


class VectorStoreError(AppError):
    def __init__(self, message: str = "Vector database operation failed."):
        super().__init__("CHROMADB_ERROR", message, 503)


class DatabaseError(AppError):
    def __init__(self, message: str = "Database operation failed."):
        super().__init__("DATABASE_ERROR", message, 503)


class InvalidTokenError(AppError):
    def __init__(self, message: str = "Could not validate credentials"):
        super().__init__("INVALID_JWT", message, 401)


class UnauthorizedDocumentError(AppError):
    def __init__(self, message: str = "You are not authorized to access this document."):
        super().__init__("UNAUTHORIZED_DOCUMENT_ACCESS", message, 403)


class EmptyRetrievalError(AppError):
    def __init__(self, message: str = "No relevant evidence was retrieved."):
        super().__init__("EMPTY_RETRIEVAL", message, 404)


class InsufficientEvidenceError(AppError):
    def __init__(self, message: str = "Insufficient evidence to answer reliably."):
        super().__init__("INSUFFICIENT_EVIDENCE", message, 422)
