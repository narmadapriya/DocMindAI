from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .metadata import ChunkMetadata, build_metadata
from .splitter import TextSplitter


@dataclass
class RetrievalChunk:
    """
    Retrieval-ready representation.

    Every modality eventually becomes this structure.
    """

    chunk_id: str
    content: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
        }


class ModalityAwareChunker:
    """
    Converts a Phase 5/6 ParsedDocument into retrieval-ready
    chunks.

    Supported modalities:

        text
        table
        image
        chart
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ):
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def chunk_document(
        self,
        document: Any,
        *,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[RetrievalChunk]:

        document_id = (
            document_id
            or self._get(document, "document_id")
            or self._get(
                self._get(document, "metadata"),
                "document_id",
            )
        )

        user_id = (
            user_id
            or self._get(document, "user_id")
            or self._get(
                self._get(document, "metadata"),
                "user_id",
            )
        )

        filename = self._document_filename(document)
        file_type = self._document_file_type(
            document,
            filename,
        )

        chunks: List[RetrievalChunk] = []

        # -----------------------------------------------------
        # TEXT
        # -----------------------------------------------------

        for text_block in self._get_list(
            document,
            "text_blocks",
        ):
            chunks.extend(
                self._chunk_text_block(
                    text_block,
                    document_id=document_id,
                    user_id=user_id,
                    filename=filename,
                    file_type=file_type,
                )
            )

        # -----------------------------------------------------
        # TABLES
        # -----------------------------------------------------

        for table in self._get_list(
            document,
            "tables",
        ):
            chunks.append(
                self._chunk_table(
                    table,
                    document_id=document_id,
                    user_id=user_id,
                    filename=filename,
                    file_type=file_type,
                )
            )

        # -----------------------------------------------------
        # IMAGES
        # -----------------------------------------------------

        for image in self._get_list(
            document,
            "images",
        ):
            chunks.append(
                self._chunk_image(
                    image,
                    document_id=document_id,
                    user_id=user_id,
                    filename=filename,
                    file_type=file_type,
                )
            )

        # -----------------------------------------------------
        # CHARTS
        # -----------------------------------------------------

        for chart in self._get_list(
            document,
            "charts",
        ):
            chunks.append(
                self._chunk_chart(
                    chart,
                    document_id=document_id,
                    user_id=user_id,
                    filename=filename,
                    file_type=file_type,
                )
            )

        # Re-index after all modalities are processed.
        for index, chunk in enumerate(chunks):
            chunk.metadata.chunk_index = index

        return chunks

    # ---------------------------------------------------------
    # TEXT
    # ---------------------------------------------------------

    def _chunk_text_block(
        self,
        block: Any,
        *,
        document_id: Optional[str],
        user_id: Optional[str],
        filename: Optional[str],
        file_type: Optional[str],
    ) -> List[RetrievalChunk]:

        text = self._get(block, "text", "")

        if not text:
            return []

        page_number = self._get(
            block,
            "page_number",
        )

        block_metadata = self._get(block, "metadata", {}) or {}

        section = self._get(block, "section")
        heading_value = self._get(block, "heading")
        if not section and isinstance(heading_value, str):
            section = heading_value
        if not section:
            section = self._get(block_metadata, "section")

        sheet_name = (
            self._get(block, "sheet_name")
            or self._get(block_metadata, "sheet_name")
        )

        row_range = (
            self._get(block, "row_range")
            or self._get(block_metadata, "row_range")
        )

        source_location = (
            self._get(block, "source_location")
            or self._get(block, "location")
            or self._get(block_metadata, "source_location")
        )

        pieces = self.splitter.split(text)

        result: List[RetrievalChunk] = []

        for index, piece in enumerate(pieces):

            content = piece

            metadata = build_metadata(
                document_id=document_id,
                user_id=user_id,
                filename=filename,
                file_type=file_type,
                page_number=page_number,
                section=section,
                chunk_type="text",
                sheet_name=sheet_name,
                row_range=row_range,
                source_location=source_location,
                chunk_index=index,
                extra={
                    "block_type": "text",
                    "parser_source": self._get(block_metadata, "source"),
                },
            )

            result.append(
                self._make_chunk(
                    content=content,
                    metadata=metadata,
                    parent_id=(
                        self._get(block, "id")
                        or self._get(block, "block_id")
                    ),
                )
            )

        return result

    # ---------------------------------------------------------
    # TABLE
    # ---------------------------------------------------------

    def _chunk_table(
        self,
        table: Any,
        *,
        document_id: Optional[str],
        user_id: Optional[str],
        filename: Optional[str],
        file_type: Optional[str],
    ) -> RetrievalChunk:

        headers = self._get(
            table,
            "headers",
            [],
        )

        rows = self._get(
            table,
            "rows",
            [],
        )

        table_id = (
            self._get(table, "table_id")
            or self._get(table, "id")
        )

        page_number = self._get(
            table,
            "page_number",
        )

        sheet_name = self._get(
            table,
            "sheet_name",
        )

        source_location = (
            self._get(table, "source_location")
            or self._get(table, "location")
        )

        row_range = self._get(
            table,
            "row_range",
        )

        content_lines: List[str] = []

        if headers:
            content_lines.append(
                " | ".join(
                    str(value)
                    for value in headers
                )
            )

        for row in rows:
            if isinstance(row, dict):
                values = [
                    str(value)
                    for value in row.values()
                ]
            elif isinstance(row, (list, tuple)):
                values = [
                    str(value)
                    for value in row
                ]
            else:
                values = [str(row)]

            content_lines.append(
                " | ".join(values)
            )

        content = "\n".join(content_lines).strip()

        # Table-aware metadata.
        metadata = build_metadata(
            document_id=document_id,
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            page_number=page_number,
            section=self._get(table, "section"),
            chunk_type="table",
            table_id=table_id,
            sheet_name=sheet_name,
            row_range=row_range,
            source_location=source_location,
            extra={
                "row_count": len(rows),
                "column_count": len(headers)
                if headers
                else self._infer_column_count(rows),
            },
        )

        return self._make_chunk(
            content=content,
            metadata=metadata,
            parent_id=table_id,
        )

    # ---------------------------------------------------------
    # IMAGE
    # ---------------------------------------------------------

    def _chunk_image(
        self,
        image: Any,
        *,
        document_id: Optional[str],
        user_id: Optional[str],
        filename: Optional[str],
        file_type: Optional[str],
    ) -> RetrievalChunk:

        image_id = (
            self._get(image, "image_id")
            or self._get(image, "id")
        )

        page_number = self._get(
            image,
            "page_number",
        )

        sheet_name = self._get(
            image,
            "sheet_name",
        )

        source_location = (
            self._get(image, "source_location")
            or self._get(image, "path")
            or self._get(image, "image_path")
            or self._get(image, "location")
        )

        ocr_text = self._get(
            image,
            "ocr_text",
        )

        visual_description = (
            self._get(image, "visual_description")
            or self._get(image, "description")
        )

        content_parts = []

        if visual_description:
            content_parts.append(
                f"Visual description:\n{visual_description}"
            )

        if ocr_text:
            content_parts.append(
                f"OCR text:\n{ocr_text}"
            )

        if not content_parts:
            content_parts.append(
                "Image content extracted from document."
            )

        content = "\n\n".join(content_parts)

        metadata = build_metadata(
            document_id=document_id,
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            page_number=page_number,
            section=self._get(image, "section"),
            chunk_type="image",
            image_id=image_id,
            sheet_name=sheet_name,
            source_location=source_location,
            ocr_text=ocr_text,
            visual_description=visual_description,
        )

        return self._make_chunk(
            content=content,
            metadata=metadata,
            parent_id=image_id,
        )

    # ---------------------------------------------------------
    # CHART
    # ---------------------------------------------------------

    def _chunk_chart(
        self,
        chart: Any,
        *,
        document_id: Optional[str],
        user_id: Optional[str],
        filename: Optional[str],
        file_type: Optional[str],
    ) -> RetrievalChunk:

        chart_id = (
            self._get(chart, "chart_id")
            or self._get(chart, "id")
        )

        page_number = self._get(
            chart,
            "page_number",
        )

        sheet_name = self._get(
            chart,
            "sheet_name",
        )

        source_location = (
            self._get(chart, "source_location")
            or self._get(chart, "path")
            or self._get(chart, "location")
        )

        description = (
            self._get(chart, "visual_description")
            or self._get(chart, "description")
        )

        chart_type = self._get(
            chart,
            "chart_type",
        )

        title = self._get(
            chart,
            "title",
        )

        axes = self._get(
            chart,
            "axes",
            [],
        )

        trends = self._get(
            chart,
            "trends",
            [],
        )

        numerical_relationships = self._get(
            chart,
            "numerical_relationships",
            [],
        )

        comparisons = self._get(
            chart,
            "comparisons",
            [],
        )

        content_parts = []

        if title:
            content_parts.append(
                f"Chart title: {title}"
            )

        if chart_type:
            content_parts.append(
                f"Chart type: {chart_type}"
            )

        if description:
            content_parts.append(
                f"Visual description:\n{description}"
            )

        if axes:
            content_parts.append(
                "Axes:\n"
                + self._stringify_list(axes)
            )

        if trends:
            content_parts.append(
                "Trends:\n"
                + self._stringify_list(trends)
            )

        if numerical_relationships:
            content_parts.append(
                "Numerical relationships:\n"
                + self._stringify_list(
                    numerical_relationships
                )
            )

        if comparisons:
            content_parts.append(
                "Comparisons:\n"
                + self._stringify_list(comparisons)
            )

        if not content_parts:
            content_parts.append(
                "Chart/graph extracted from document."
            )

        content = "\n\n".join(content_parts)

        metadata = build_metadata(
            document_id=document_id,
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            page_number=page_number,
            section=self._get(chart, "section"),
            chunk_type="chart",
            image_id=chart_id,
            sheet_name=sheet_name,
            source_location=source_location,
            visual_description=description,
            extra={
                "chart_type": chart_type,
            },
        )

        return self._make_chunk(
            content=content,
            metadata=metadata,
            parent_id=chart_id,
        )

    # ---------------------------------------------------------
    # CHUNK ID
    # ---------------------------------------------------------

    def _make_chunk(
        self,
        *,
        content: str,
        metadata: ChunkMetadata,
        parent_id: Optional[str] = None,
    ) -> RetrievalChunk:

        metadata.parent_id = parent_id

        identity = {
            "document_id": metadata.document_id,
            "chunk_type": metadata.chunk_type,
            "table_id": metadata.table_id,
            "image_id": metadata.image_id,
            "page_number": metadata.page_number,
            "sheet_name": metadata.sheet_name,
            "source_location": metadata.source_location,
            "content": content,
        }

        digest = hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:24]

        chunk_id = f"chunk_{digest}"

        return RetrievalChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
        )

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _get(
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:

        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(key, default)

        return getattr(obj, key, default)

    @classmethod
    def _get_list(
        cls,
        obj: Any,
        key: str,
    ) -> List[Any]:

        value = cls._get(obj, key, [])

        if value is None:
            return []

        if isinstance(value, Iterable) and not isinstance(
            value,
            (str, bytes, dict),
        ):
            return list(value)

        return []

    @classmethod
    def _document_filename(
        cls,
        document: Any,
    ) -> Optional[str]:

        return (
            cls._get(document, "filename")
            or cls._get(document, "file_name")
            or cls._get(
                cls._get(document, "metadata"),
                "filename",
            )
        )

    @classmethod
    def _document_file_type(
        cls,
        document: Any,
        filename: Optional[str],
    ) -> Optional[str]:

        file_type = (
            cls._get(document, "file_type")
            or cls._get(document, "mime_type")
            or cls._get(
                cls._get(document, "metadata"),
                "file_type",
            )
        )

        if file_type:
            return str(file_type).upper()

        if filename and "." in filename:
            return filename.rsplit(
                ".",
                1,
            )[-1].upper()

        return None

    @staticmethod
    def _infer_column_count(
        rows: List[Any],
    ) -> int:

        if not rows:
            return 0

        first = rows[0]

        if isinstance(first, dict):
            return len(first)

        if isinstance(first, (list, tuple)):
            return len(first)

        return 1

    @staticmethod
    def _stringify_list(
        values: Any,
    ) -> str:

        if isinstance(values, str):
            return values

        if not isinstance(values, (list, tuple)):
            return str(values)

        return "\n".join(
            f"- {value}"
            for value in values
        )


# Convenient function for pipeline usage.
def chunk_document(
    document: Any,
    *,
    document_id: Optional[str] = None,
    user_id: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[RetrievalChunk]:

    chunker = ModalityAwareChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunker.chunk_document(
        document,
        document_id=document_id,
        user_id=user_id,
    )