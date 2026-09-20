"""DOCX parser: paragraphs/headings, tables, embedded images and metadata."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from zipfile import ZipFile

from app.rag.parser.exceptions import DocumentExtractionError
from app.rag.parser.models import (
    ChartBlock,
    DocumentMetadata,
    ImageBlock,
    ParsedDocument,
    TableBlock,
    TextBlock,
)


_MAX_SECTION_CHARS = 2600


def _style_name(paragraph: object) -> str:
    style = getattr(paragraph, "style", None)
    return str(getattr(style, "name", "") or "").strip()


def _table_value(value: str, label: str) -> str:
    cleaned = str(value or "").strip()
    if "revenue" in str(label).lower():
        numeric = cleaned.replace("$", "").replace(",", "").strip()
        try:
            amount = Decimal(numeric)
        except InvalidOperation:
            amount = None

        if amount is not None and abs(amount) >= Decimal("1000000"):
            millions = (amount / Decimal("1000000")).normalize()
            million_text = format(millions, "f").rstrip("0").rstrip(".")
            return f"{cleaned} ({million_text} million USD)"
    return cleaned


def _table_text(table_index: int, rows: list[list[str]]) -> str:
    """Build a compact labeled text mirror of a DOCX table for text retrieval."""
    if not rows:
        return ""

    headers = [str(cell).strip() or f"Column {index + 1}" for index, cell in enumerate(rows[0])]
    lines = [f"DOCX table {table_index}.", "Columns: " + ", ".join(headers)]

    for row_number, row in enumerate(rows[1:], start=2):
        values = list(row) + [""] * max(0, len(headers) - len(row))
        fields = [
            f"{headers[index]}: {_table_value(values[index], headers[index])}"
            for index in range(min(len(headers), len(values)))
            if str(values[index]).strip()
        ]
        if fields:
            lines.append(f"Row {row_number}: " + "; ".join(fields))

    # A one-row table has no data rows; retain its raw values.
    if len(rows) == 1:
        lines.append(" | ".join(str(value).strip() for value in rows[0]))

    return "\n".join(lines).strip()


def _paragraph_text_blocks(doc: object) -> list[TextBlock]:
    """
    Build retrieval-oriented DOCX text blocks.

    DOCX stores headings and each body paragraph separately.  Keeping each
    paragraph as an independent embedding fragments one logical section across
    several chunks.  PDF pages do not have that problem, which is why the same
    Chat & Ask question can work for PDF but fail verification for DOCX.

    This function preserves heading metadata while grouping consecutive body
    paragraphs under the same heading into one bounded semantic block.  It does
    not change the source document or any downstream RAG/API behavior.
    """

    blocks: list[TextBlock] = []
    active_heading: str | None = None
    body_parts: list[str] = []
    body_styles: list[str] = []
    first_order = 0

    def flush_body() -> None:
        nonlocal body_parts, body_styles, first_order
        if not body_parts:
            return

        # Keep each semantic block bounded so the existing TextSplitter still
        # controls final chunk size without feeding Qwen unnecessarily huge text.
        current: list[str] = []
        current_chars = 0

        def append_current() -> None:
            nonlocal current, current_chars
            if not current:
                return
            body = "\n".join(current).strip()
            block_text = f"{active_heading}\n{body}" if active_heading else body
            blocks.append(
                TextBlock(
                    text=block_text,
                    heading=False,
                    section=active_heading,
                    order=first_order + len(blocks),
                    metadata={
                        "style": body_styles[0] if body_styles else "",
                        "source": "docx_section_body" if active_heading else "docx_paragraph_group",
                        "heading_context": active_heading,
                        "paragraph_count": len(current),
                    },
                )
            )
            current = []
            current_chars = 0

        for part in body_parts:
            projected = current_chars + len(part) + (1 if current else 0)
            if current and projected > _MAX_SECTION_CHARS:
                append_current()
            current.append(part)
            current_chars += len(part) + (1 if current_chars else 0)

        append_current()
        body_parts = []
        body_styles = []

    for index, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text.strip()
        if not text:
            continue

        style_name = _style_name(paragraph)
        is_heading = style_name.lower().startswith("heading")

        if is_heading:
            flush_body()
            active_heading = text
            first_order = index
            # Preserve the heading itself as a small searchable block for
            # navigation/section-name queries.
            blocks.append(
                TextBlock(
                    text=text,
                    heading=True,
                    section=text,
                    order=index,
                    metadata={
                        "style": style_name,
                        "source": "docx_heading",
                        "heading_context": text,
                    },
                )
            )
            continue

        if not body_parts:
            first_order = index
        body_parts.append(text)
        body_styles.append(style_name)

    flush_body()
    return blocks


def parse_docx(file_path: str | Path, output_dir: str | Path | None = None) -> ParsedDocument:
    """Parse a DOCX into DocMindAI's canonical ParsedDocument."""

    path = Path(file_path)
    if not path.exists():
        raise DocumentExtractionError(f"DOCX not found: {path}")

    try:
        from docx import Document
    except ImportError as exc:
        raise DocumentExtractionError("python-docx is required for DOCX parsing.") from exc

    output = Path(output_dir) if output_dir else path.parent / "parsed_assets" / path.stem
    output.mkdir(parents=True, exist_ok=True)

    try:
        doc = Document(path)
        core = doc.core_properties

        metadata = DocumentMetadata(
            filename=path.name,
            file_type=".docx",
            file_size=path.stat().st_size,
            title=core.title or None,
            author=core.author or None,
            subject=core.subject or None,
            creator=core.author or None,
            created_at=core.created.isoformat() if core.created else None,
            modified_at=core.modified.isoformat() if core.modified else None,
            extra={"last_modified_by": core.last_modified_by},
        )

        text_blocks = _paragraph_text_blocks(doc)

        tables: list[TableBlock] = []
        table_text_blocks: list[TextBlock] = []
        for table_index, table in enumerate(doc.tables, start=1):
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            rows = [row for row in rows if any(cell for cell in row)]
            if not rows:
                continue

            table_id = f"docx-table{table_index}"
            tables.append(
                TableBlock(
                    rows=rows,
                    headers=rows[0] if len(rows) > 1 else None,
                    table_id=table_id,
                    order=table_index - 1,
                    metadata={"source": "docx_table"},
                )
            )

            # A text mirror lets ordinary natural-language questions retrieve
            # table facts even when the router's preferred modality is "text".
            # The original TableBlock is preserved unchanged for table-aware RAG.
            serialized = _table_text(table_index, rows)
            if serialized:
                table_text_blocks.append(
                    TextBlock(
                        text=serialized,
                        order=len(text_blocks) + len(table_text_blocks),
                        metadata={
                            "source": "docx_table_text",
                            "table_id": table_id,
                            "row_count": len(rows),
                        },
                    )
                )

        text_blocks.extend(table_text_blocks)

        images: list[ImageBlock] = []
        with ZipFile(path) as archive:
            media_files = [name for name in archive.namelist() if name.startswith("word/media/")]
            for image_index, member in enumerate(media_files, start=1):
                image_name = Path(member).name
                image_path = output / image_name
                image_path.write_bytes(archive.read(member))
                suffix = image_path.suffix.lower().lstrip(".") or "png"
                images.append(
                    ImageBlock(
                        path=str(image_path),
                        image_id=f"docx-img{image_index}",
                        mime_type=f"image/{suffix}",
                        order=image_index - 1,
                        metadata={
                            "archive_member": member,
                            "needs_vision": True,
                            "source": "docx_embedded_image",
                        },
                    )
                )

        charts = [
            ChartBlock(
                chart_id=f"{image.image_id}-visual",
                image_path=image.path,
                order=index,
                metadata={
                    "candidate": True,
                    "needs_vision_classification": True,
                    "source_image_id": image.image_id,
                },
            )
            for index, image in enumerate(images)
        ]

        return ParsedDocument(
            metadata=metadata,
            text_blocks=text_blocks,
            tables=tables,
            images=images,
            charts=charts,
        )
    except DocumentExtractionError:
        raise
    except Exception as exc:
        raise DocumentExtractionError(f"Failed to parse DOCX '{path.name}': {exc}") from exc
