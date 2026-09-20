"""Plain TXT parser."""

from pathlib import Path

from app.rag.parser.exceptions import DocumentExtractionError
from app.rag.parser.models import DocumentMetadata, ParsedDocument, TextBlock


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentExtractionError(f"Unable to decode text file: {path.name}")


def parse_txt(file_path: str | Path) -> ParsedDocument:
    path = Path(file_path)
    if not path.exists():
        raise DocumentExtractionError(f"TXT not found: {path}")
    try:
        text = _read_text(path).strip()
        blocks = []
        if text:
            blocks.append(
                TextBlock(
                    text=text,
                    order=0,
                    metadata={"source": "txt"},
                )
            )
        return ParsedDocument(
            metadata=DocumentMetadata(
                filename=path.name,
                file_type=".txt",
                file_size=path.stat().st_size,
            ),
            text_blocks=blocks,
        )
    except DocumentExtractionError:
        raise
    except Exception as exc:
        raise DocumentExtractionError(f"Failed to parse TXT '{path.name}': {exc}") from exc
