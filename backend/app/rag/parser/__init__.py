"""Phase 5 unified document parsing API."""

from app.rag.parser.exceptions import (
    DocumentExtractionError,
    DocumentParsingError,
    UnsupportedDocumentTypeError,
)
from app.rag.parser.loader import PARSERS, SUPPORTED_EXTENSIONS, is_supported, parse_document
from app.rag.parser.models import (
    ChartBlock,
    DocumentMetadata,
    ImageBlock,
    ParsedDocument,
    TableBlock,
    TextBlock,
)

__all__ = [
    "PARSERS",
    "SUPPORTED_EXTENSIONS",
    "is_supported",
    "parse_document",
    "DocumentParsingError",
    "DocumentExtractionError",
    "UnsupportedDocumentTypeError",
    "ParsedDocument",
    "DocumentMetadata",
    "TextBlock",
    "TableBlock",
    "ImageBlock",
    "ChartBlock",
]
