"""PDF parser: text, tables, embedded images and metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.rag.parser.exceptions import DocumentExtractionError
from app.rag.parser.models import (
    ChartBlock,
    DocumentMetadata,
    ImageBlock,
    ParsedDocument,
    TableBlock,
    TextBlock,
)


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _image_extension(image: dict[str, Any]) -> str:
    ext = image.get("ext") or "png"
    return ".jpg" if ext.lower() in {"jpeg", "jpe"} else f".{ext.lower()}"


def parse_pdf(file_path: str | Path, output_dir: str | Path | None = None) -> ParsedDocument:
    path = Path(file_path)
    if not path.exists():
        raise DocumentExtractionError(f"PDF not found: {path}")

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise DocumentExtractionError("PyMuPDF is required for PDF parsing.") from exc

    try:
        import pdfplumber
    except ImportError:
        pdfplumber = None

    output = Path(output_dir) if output_dir else path.parent / "parsed_assets" / path.stem
    output.mkdir(parents=True, exist_ok=True)

    text_blocks: list[TextBlock] = []
    tables: list[TableBlock] = []
    images: list[ImageBlock] = []
    charts: list[ChartBlock] = []

    try:
        document = fitz.open(path)
        metadata = document.metadata or {}

        doc_metadata = DocumentMetadata(
            filename=path.name,
            file_type=".pdf",
            file_size=path.stat().st_size,
            title=metadata.get("title") or None,
            author=metadata.get("author") or None,
            subject=metadata.get("subject") or None,
            creator=metadata.get("creator") or None,
            page_count=len(document),
            extra={"format": metadata.get("format")},
        )

        text_order = image_order = 0
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            if text.strip():
                text_blocks.append(
                    TextBlock(
                        text=text.strip(),
                        page_number=page_index,
                        order=text_order,
                        metadata={"source": "pdf_text"},
                    )
                )
                text_order += 1

            for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                extracted = document.extract_image(xref)
                image_ext = extracted.get("ext", "png")
                image_name = f"page_{page_index}_image_{image_index}.{image_ext}"
                image_path = output / image_name
                image_path.write_bytes(extracted["image"])
                images.append(
                    ImageBlock(
                        path=str(image_path),
                        page_number=page_index,
                        image_id=f"pdf-p{page_index}-img{image_index}",
                        mime_type=f"image/{image_ext.lower()}",
                        width=extracted.get("width"),
                        height=extracted.get("height"),
                        order=image_order,
                        metadata={
                            "xref": xref,
                            "needs_vision": True,
                            "source": "pdf_embedded_image",
                        },
                    )
                )
                image_order += 1

        document.close()

        if pdfplumber is not None:
            with pdfplumber.open(path) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    extracted_tables = page.extract_tables() or []
                    for table_index, raw_table in enumerate(extracted_tables, start=1):
                        rows = [[_clean(cell) for cell in row] for row in raw_table]
                        rows = [row for row in rows if any(cell for cell in row)]
                        if not rows:
                            continue
                        headers = rows[0] if len(rows) > 1 else None
                        tables.append(
                            TableBlock(
                                rows=rows,
                                headers=headers,
                                page_number=page_index,
                                table_id=f"pdf-p{page_index}-table{table_index}",
                                order=len(tables),
                                metadata={"source": "pdf_table"},
                            )
                        )

        # PDF charts are generally embedded as images. Phase 5 identifies those
        # visual regions; Phase 6 sends the image bytes to Qwen2.5-VL.
        for image in images:
            charts.append(
                ChartBlock(
                    chart_id=f"{image.image_id}-visual",
                    page_number=image.page_number,
                    image_path=image.path,
                    order=len(charts),
                    metadata={
                        "candidate": True,
                        "needs_vision_classification": True,
                        "source_image_id": image.image_id,
                    },
                )
            )

        return ParsedDocument(
            metadata=doc_metadata,
            text_blocks=text_blocks,
            tables=tables,
            images=images,
            charts=charts,
        )
    except DocumentExtractionError:
        raise
    except Exception as exc:
        raise DocumentExtractionError(f"Failed to parse PDF '{path.name}': {exc}") from exc
