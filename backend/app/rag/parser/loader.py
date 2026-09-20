"""Unified document parser dispatcher for the five supported formats."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.rag.parser.csv_parser import parse_csv
from app.rag.parser.docx_parser import parse_docx
from app.rag.parser.exceptions import UnsupportedDocumentTypeError
from app.rag.parser.excel_parser import parse_xlsx
from app.rag.parser.models import ParsedDocument
from app.rag.parser.pdf_parser import parse_pdf
from app.rag.parser.txt_parser import parse_txt

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx"}

Parser = Callable[..., ParsedDocument]

PARSERS: dict[str, Parser] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_txt,
    ".csv": parse_csv,
    ".xlsx": parse_xlsx,
}


def get_extension(file_path: str | Path) -> str:
    return Path(file_path).suffix.lower()


def is_supported(file_path: str | Path) -> bool:
    return get_extension(file_path) in SUPPORTED_EXTENSIONS


def parse_document(
    file_path: str | Path,
    output_dir: str | Path | None = None,
) -> ParsedDocument:
    path = Path(file_path)
    extension = get_extension(path)

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentTypeError(
            f"Unsupported file type '{extension or '[no extension]'}'. "
            "Supported formats: PDF, DOCX, TXT, CSV, XLSX."
        )

    parser = PARSERS[extension]
    if extension in {".pdf", ".docx", ".xlsx"}:
        return parser(path, output_dir=output_dir)
    return parser(path)
