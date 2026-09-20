class DocumentParsingError(Exception):
    """Base error for Phase 5 document parsing."""


class UnsupportedDocumentTypeError(DocumentParsingError):
    """Raised when a file extension is not supported by DocMindAI."""


class DocumentExtractionError(DocumentParsingError):
    """Raised when a supported document cannot be extracted."""
