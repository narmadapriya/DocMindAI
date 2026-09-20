"""Shared image metadata helper used by document parsers.

Actual multimodal interpretation is intentionally deferred to Phase 6. Phase 5
only records the visual asset so it can later be sent to Qwen2.5-VL.
"""

from __future__ import annotations

from pathlib import Path

from app.rag.parser.models import ImageBlock


def build_image_block(
    path: str | Path,
    *,
    image_id: str,
    page_number: int | None = None,
    sheet_name: str | None = None,
    mime_type: str | None = None,
    width: int | None = None,
    height: int | None = None,
    source: str = "unknown",
    **metadata,
) -> ImageBlock:
    return ImageBlock(
        path=str(path),
        page_number=page_number,
        sheet_name=sheet_name,
        image_id=image_id,
        mime_type=mime_type,
        width=width,
        height=height,
        metadata={
            "needs_vision": True,
            "source": source,
            **metadata,
        },
    )
