from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ParsedDocument:
    """
    Unified internal representation of a parsed document.

    This is intentionally independent of PostgreSQL.

    It is the common internal representation used by
    the document ingestion and future RAG pipeline.

    Supported modalities:
        - Text
        - Tables
        - Images
        - Charts
        - Metadata
        - Page information

    This class does not perform parsing.
    It only represents normalized parser output.
    """

    document_id: int | None = None

    filename: str = ""

    file_type: str = ""

    text_blocks: List[Dict[str, Any]] = field(
        default_factory=list
    )

    tables: List[Dict[str, Any]] = field(
        default_factory=list
    )

    images: List[Dict[str, Any]] = field(
        default_factory=list
    )

    charts: List[Dict[str, Any]] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    raw: Dict[str, Any] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # TEXT
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        """
        Return all text blocks as one string.

        Each normalized text block is expected to contain:

            {
                "text": "..."
            }

        Empty or malformed blocks are ignored.
        """

        parts: List[str] = []

        for block in self.text_blocks:

            if not isinstance(block, dict):
                continue

            value = block.get("text")

            if value:
                parts.append(str(value))

        return "\n\n".join(parts)

    @property
    def text_count(self) -> int:
        """
        Number of normalized text blocks.
        """

        return len(self.text_blocks)

    @property
    def has_text(self) -> bool:
        """
        True when the document contains usable text.
        """

        return bool(self.text.strip())

    # ------------------------------------------------------------------
    # TABLES
    # ------------------------------------------------------------------

    @property
    def table_count(self) -> int:
        """
        Number of extracted tables.
        """

        return len(self.tables)

    @property
    def has_tables(self) -> bool:
        """
        True when the document contains at least one table.
        """

        return bool(self.tables)

    # ------------------------------------------------------------------
    # IMAGES
    # ------------------------------------------------------------------

    @property
    def image_count(self) -> int:
        """
        Number of extracted images.
        """

        return len(self.images)

    @property
    def has_images(self) -> bool:
        """
        True when the document contains at least one image.
        """

        return bool(self.images)

    # ------------------------------------------------------------------
    # CHARTS
    # ------------------------------------------------------------------

    @property
    def chart_count(self) -> int:
        """
        Number of extracted charts.
        """

        return len(self.charts)

    @property
    def has_charts(self) -> bool:
        """
        True when the document contains at least one chart.
        """

        return bool(self.charts)

    # ------------------------------------------------------------------
    # PAGES
    # ------------------------------------------------------------------

    @property
    def page_count(self) -> int:
        """
        Return the number of pages represented by the document.

        Priority:

        1. Explicit metadata["page_count"]
        2. Number of unique page values found in text blocks
        3. Number of pages in raw["pages"]
        4. 0 when page information is unavailable
        """

        # --------------------------------------------------------------
        # 1. Explicit page count from normalized metadata
        # --------------------------------------------------------------

        metadata_page_count = self.metadata.get(
            "page_count"
        )

        if isinstance(metadata_page_count, int):
            return max(metadata_page_count, 0)

        if isinstance(metadata_page_count, str):

            try:
                return max(
                    int(metadata_page_count),
                    0,
                )
            except ValueError:
                pass

        # --------------------------------------------------------------
        # 2. Infer page count from text blocks
        # --------------------------------------------------------------

        pages = set()

        for block in self.text_blocks:

            if not isinstance(block, dict):
                continue

            page = block.get("page")

            if page is not None:
                pages.add(str(page))

        if pages:
            return len(pages)

        # --------------------------------------------------------------
        # 3. Infer page count from raw parser output
        # --------------------------------------------------------------

        raw_pages = self.raw.get("pages")

        if isinstance(raw_pages, list):
            return len(raw_pages)

        # --------------------------------------------------------------
        # 4. No page information available
        # --------------------------------------------------------------

        return 0

    # ------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the unified representation into a serializable dict.

        The original normalized data is preserved.

        Count properties are intentionally not stored separately because
        they are derived from the modality collections.
        """

        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "text_blocks": self.text_blocks,
            "tables": self.tables,
            "images": self.images,
            "charts": self.charts,
            "metadata": self.metadata,
        }

