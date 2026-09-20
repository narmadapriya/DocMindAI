from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ============================================================
# Text Block
# ============================================================

@dataclass
class TextBlock:
    """
    Unified representation of extracted text.

    Used for:
        - Paragraphs
        - Headings
        - Sections
        - Page text
        - Plain text
    """

    text: str

    page_number: int | None = None
    section: str | None = None
    heading: bool = False
    order: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Table Block
# ============================================================

@dataclass
class TableBlock:
    """
    Unified representation of an extracted table.

    Supports:
        - PDF tables
        - DOCX tables
        - CSV data
        - XLSX worksheets/tables
    """

    rows: list[list[str]]

    page_number: int | None = None
    sheet_name: str | None = None
    table_id: str | None = None
    title: str | None = None

    headers: list[str] | None = None

    order: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Image Block
# ============================================================

@dataclass
class ImageBlock:
    """
    Unified representation of an extracted image.

    The physical image remains stored on disk.
    This object stores image metadata and provenance.
    """

    path: str

    page_number: int | None = None
    sheet_name: str | None = None

    image_id: str | None = None
    mime_type: str | None = None

    width: int | None = None
    height: int | None = None

    order: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Chart Block
# ============================================================

@dataclass
class ChartBlock:
    """
    Unified representation of a chart or graph.

    Charts can originate from:
        - PDF visual regions
        - DOCX images/visuals
        - XLSX charts
    """

    chart_id: str

    page_number: int | None = None
    sheet_name: str | None = None

    chart_type: str | None = None
    title: str | None = None

    image_path: str | None = None
    source_range: str | None = None

    order: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Document Metadata
# ============================================================

@dataclass
class DocumentMetadata:
    """
    Unified document-level metadata.

    Works across:
        PDF
        DOCX
        TXT
        CSV
        XLSX
    """

    filename: str
    file_type: str
    file_size: int

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creator: str | None = None

    created_at: str | None = None
    modified_at: str | None = None

    page_count: int | None = None

    sheet_names: list[str] = field(
        default_factory=list
    )

    extra: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Unified Parsed Document
# ============================================================

@dataclass
class ParsedDocument:
    """
    Canonical internal document representation.

    Every supported parser returns this structure.

    Structure:

        ParsedDocument
        │
        ├── Metadata
        ├── Text Blocks
        ├── Tables
        ├── Images
        ├── Charts
        └── Visual Evidence

    This representation is independent from:
        - PostgreSQL
        - FastAPI
        - ChromaDB
        - Embedding models
        - RAG agents
    """

    metadata: DocumentMetadata

    text_blocks: list[TextBlock] = field(
        default_factory=list
    )

    tables: list[TableBlock] = field(
        default_factory=list
    )

    images: list[ImageBlock] = field(
        default_factory=list
    )

    charts: list[ChartBlock] = field(
        default_factory=list
    )

    visual_evidence: list[dict[str, Any]] = field(
        default_factory=list
    )

    # ========================================================
    # Serialization
    # ========================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the entire document representation into
        a serializable dictionary.
        """

        data = asdict(self)

        # Preserve backward compatibility:
        # omit visual_evidence when unused.
        if not self.visual_evidence:
            data.pop(
                "visual_evidence",
                None,
            )

        return data

    # ========================================================
    # Document Identity
    # ========================================================

    @property
    def filename(self) -> str:
        """
        Return original document filename.
        """

        return self.metadata.filename

    @property
    def file_type(self) -> str:
        """
        Return normalized document extension.
        """

        return self.metadata.file_type

    @property
    def file_size(self) -> int:
        """
        Return document size in bytes.
        """

        return self.metadata.file_size

    # ========================================================
    # Text
    # ========================================================

    @property
    def text(self) -> str:
        """
        Combine all extracted text blocks into one text stream.
        """

        return "\n\n".join(
            block.text
            for block in self.text_blocks
            if block.text.strip()
        )

    @property
    def has_text(self) -> bool:
        """
        True when usable text exists.
        """

        return bool(
            self.text.strip()
        )

    @property
    def text_block_count(self) -> int:
        """
        Number of extracted text blocks.
        """

        return len(
            self.text_blocks
        )

    # ========================================================
    # Tables
    # ========================================================

    @property
    def has_tables(self) -> bool:
        """
        True when one or more tables exist.
        """

        return bool(
            self.tables
        )

    @property
    def table_count(self) -> int:
        """
        Number of extracted tables.
        """

        return len(
            self.tables
        )

    # ========================================================
    # Images
    # ========================================================

    @property
    def has_images(self) -> bool:
        """
        True when one or more images exist.
        """

        return bool(
            self.images
        )

    @property
    def image_count(self) -> int:
        """
        Number of extracted images.
        """

        return len(
            self.images
        )

    # ========================================================
    # Charts
    # ========================================================

    @property
    def has_charts(self) -> bool:
        """
        True when one or more charts exist.
        """

        return bool(
            self.charts
        )

    @property
    def chart_count(self) -> int:
        """
        Number of extracted charts.
        """

        return len(
            self.charts
        )

    # ========================================================
    # Page Information
    # ========================================================

    @property
    def page_count(self) -> int | None:
        """
        Return parser-provided document page count.

        TXT/CSV/XLSX may not have page information,
        therefore None is valid.
        """

        return self.metadata.page_count

    # ========================================================
    # Spreadsheet Information
    # ========================================================

    @property
    def sheet_names(self) -> list[str]:
        """
        Return spreadsheet worksheet names.

        Empty for non-spreadsheet documents.
        """

        return self.metadata.sheet_names

    # ========================================================
    # Modality Counts
    # ========================================================

    @property
    def counts(self) -> dict[str, int]:
        """
        Return counts for every unified modality.

        Useful for:
            - ingestion diagnostics
            - logging
            - testing
            - future chunking pipeline
        """

        return {
            "text_blocks": len(
                self.text_blocks
            ),
            "tables": len(
                self.tables
            ),
            "images": len(
                self.images
            ),
            "charts": len(
                self.charts
            ),
        }