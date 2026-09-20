from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class ChunkMetadata:
    """
    Retrieval metadata propagated from the original document
    and its multimodal content.
    """

    document_id: Optional[str] = None
    user_id: Optional[str] = None

    filename: Optional[str] = None
    file_type: Optional[str] = None

    page_number: Optional[int] = None
    section: Optional[str] = None

    chunk_type: Optional[str] = None

    table_id: Optional[str] = None
    image_id: Optional[str] = None

    sheet_name: Optional[str] = None
    row_range: Optional[str] = None

    source_location: Optional[str] = None

    # Additional information useful for later retrieval/citation.
    chunk_index: Optional[int] = None
    parent_id: Optional[str] = None

    # Phase 6 information.
    ocr_text: Optional[str] = None
    visual_description: Optional[str] = None

    # Arbitrary additional metadata from the parser.
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata into a plain dictionary suitable for
        ChromaDB metadata or API responses.
        """
        data = asdict(self)

        # ChromaDB metadata should not receive None values.
        return {
            key: value
            for key, value in data.items()
            if value is not None
        }


def normalize_file_type(filename: Optional[str]) -> Optional[str]:
    """
    Normalize a filename extension.

    Supported project formats:
        PDF, DOCX, TXT, CSV, XLSX
    """
    if not filename:
        return None

    name = str(filename).lower()

    if "." not in name:
        return None

    return name.rsplit(".", 1)[-1].upper()


def build_metadata(
    *,
    document_id: Optional[str] = None,
    user_id: Optional[str] = None,
    filename: Optional[str] = None,
    file_type: Optional[str] = None,
    page_number: Optional[int] = None,
    section: Optional[str] = None,
    chunk_type: Optional[str] = None,
    table_id: Optional[str] = None,
    image_id: Optional[str] = None,
    sheet_name: Optional[str] = None,
    row_range: Optional[str] = None,
    source_location: Optional[str] = None,
    chunk_index: Optional[int] = None,
    parent_id: Optional[str] = None,
    ocr_text: Optional[str] = None,
    visual_description: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> ChunkMetadata:

    if file_type is None:
        file_type = normalize_file_type(filename)

    return ChunkMetadata(
        document_id=document_id,
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        page_number=page_number,
        section=section,
        chunk_type=chunk_type,
        table_id=table_id,
        image_id=image_id,
        sheet_name=sheet_name,
        row_range=row_range,
        source_location=source_location,
        chunk_index=chunk_index,
        parent_id=parent_id,
        ocr_text=ocr_text,
        visual_description=visual_description,
        extra=extra,
    )