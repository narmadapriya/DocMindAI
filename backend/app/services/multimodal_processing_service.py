from __future__ import annotations

from pathlib import Path
from typing import Any

from app.rag.llm.vision_processor import (
    OllamaVisionClient,
    VisionModelError,
)
from app.rag.ocr.ocr_service import OCRService
from app.schemas.parsed_document import ParsedDocument


class MultimodalProcessingService:
    """
    Step 6 application-level multimodal integration service.

    It connects the unified Step 3 ParsedDocument representation
    to the existing Phase 6 OCR + Qwen2.5-VL/Ollama pipeline.

    Processing strategy
    -------------------

    TEXT
        Already handled by parsers.
        No multimodal processing required.

    TABLE
        Already extracted as structured rows/headers.
        No vision-model request is required.

        We preserve a deterministic textual representation that
        can later be embedded by the existing chunking pipeline.

    IMAGE
        OCR
            +
        Qwen2.5-VL visual understanding

    CHART
        Qwen2.5-VL when an actual rendered/extracted image path
        exists.

        Native spreadsheet charts without a rendered image are
        marked "rendering_required" rather than causing ingestion
        failure.

    Design goals
    ------------
    - Preserve Steps 1-5.
    - Never fail entire ingestion because vision is unavailable.
    - Reuse existing OllamaVisionClient.
    - Reuse existing OCRService.
    - Avoid duplicate Qwen calls for the same image path.
    """

    def __init__(
        self,
        *,
        ocr_service: OCRService | None = None,
        vision_client: OllamaVisionClient | None = None,
    ):
        self.ocr = (
            ocr_service
            or OCRService()
        )

        self.vision = (
            vision_client
            or OllamaVisionClient()
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def process_document(
        self,
        document: ParsedDocument,
    ) -> ParsedDocument:
        """
        Enrich a unified ParsedDocument with multimodal evidence.

        The original object is updated in place and returned.

        This keeps the ingestion pipeline memory-efficient and
        preserves the Step 3 document representation.
        """

        if document is None:
            raise ValueError(
                "document is required."
            )

        visual_evidence: list[
            dict[str, Any]
        ] = []

        # -----------------------------------------------------
        # Cache Qwen results by source image path.
        #
        # PDF parsers may represent one visual both as:
        #   image
        #   chart candidate
        #
        # The same physical image must not be sent to Qwen twice.
        # -----------------------------------------------------

        vision_cache: dict[
            str,
            dict[str, Any]
        ] = {}

        # =====================================================
        # TABLES
        # =====================================================

        for table in document.tables:

            result = self._process_table(
                table
            )

            visual_evidence.append(
                result
            )

        # =====================================================
        # IMAGES
        # =====================================================

        for image in document.images:

            result = self._process_image(
                image,
                vision_cache=vision_cache,
            )

            visual_evidence.append(
                result
            )

        # =====================================================
        # CHARTS
        # =====================================================

        for chart in document.charts:

            result = self._process_chart(
                chart,
                vision_cache=vision_cache,
            )

            visual_evidence.append(
                result
            )

        # -----------------------------------------------------
        # Preserve evidence in the application schema.
        #
        # ParsedDocument currently contains "raw", so we store
        # document-level multimodal evidence there instead of
        # changing the frozen Step 3 schema.
        # -----------------------------------------------------

        document.raw.setdefault(
            "visual_evidence",
            visual_evidence,
        )

        document.metadata[
            "multimodal_processed"
        ] = True

        document.metadata[
            "multimodal_evidence_count"
        ] = len(
            visual_evidence
        )

        document.metadata[
            "multimodal_summary"
        ] = {
            "tables": len(
                document.tables
            ),
            "images": len(
                document.images
            ),
            "charts": len(
                document.charts
            ),
        }

        return document

    # =========================================================
    # TABLE PROCESSING
    # =========================================================

    def _process_table(
        self,
        table: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process an already structured table.

        Structured tables do NOT need Qwen2.5-VL.

        Their semantic representation is generated
        deterministically from headers + rows.
        """

        if not isinstance(
            table,
            dict,
        ):
            return {
                "asset_type": "table",
                "status": "invalid_table",
            }

        headers = table.get(
            "headers"
        ) or []

        rows = table.get(
            "rows"
        ) or []

        table_id = (
            table.get("table_id")
            or table.get("id")
        )

        page_number = table.get(
            "page_number"
        )

        sheet_name = table.get(
            "sheet_name"
        )

        table_text = (
            self._table_to_text(
                headers=headers,
                rows=rows,
            )
        )

        # Preserve representation for downstream retrieval.
        table[
            "structured_text"
        ] = table_text

        table[
            "processing_status"
        ] = "structured_ready"

        table[
            "requires_vision"
        ] = False

        return {
            "asset_id": table_id,
            "asset_type": "table",
            "page_number": page_number,
            "sheet_name": sheet_name,
            "status": "structured_ready",
            "requires_vision": False,
            "row_count": len(rows),
            "structured_text": table_text,
        }

    # =========================================================
    # IMAGE PROCESSING
    # =========================================================

    def _process_image(
        self,
        image: dict[str, Any],
        *,
        vision_cache: dict[
            str,
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        OCR + Qwen2.5-VL processing for extracted images.
        """

        if not isinstance(
            image,
            dict,
        ):
            return {
                "asset_type": "image",
                "status": "invalid_image",
            }

        image_id = (
            image.get("image_id")
            or image.get("id")
        )

        source_path = (
            image.get("path")
            or image.get("image_path")
            or image.get(
                "source_location"
            )
        )

        page_number = image.get(
            "page_number"
        )

        sheet_name = image.get(
            "sheet_name"
        )

        metadata = image.get(
            "metadata"
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

            image[
                "metadata"
            ] = metadata

        result: dict[str, Any] = {
            "asset_id": image_id,
            "asset_type": "image",
            "source_path": source_path,
            "page_number": page_number,
            "sheet_name": sheet_name,
            "ocr": None,
            "vision": None,
        }

        # -----------------------------------------------------
        # Missing physical asset.
        # -----------------------------------------------------

        if (
            not source_path
            or not Path(
                source_path
            ).exists()
        ):

            result[
                "status"
            ] = "asset_missing"

            metadata[
                "processing_status"
            ] = "asset_missing"

            return result

        # -----------------------------------------------------
        # OCR
        #
        # OCR failure must never terminate ingestion.
        # safe_extract() already implements graceful fallback.
        # -----------------------------------------------------

        try:

            ocr_result = (
                self.ocr.safe_extract(
                    source_path
                )
            )

        except Exception as exc:

            ocr_result = {
                "text": "",
                "confidence": None,
                "status": "error",
                "error": str(exc),
            }

        result[
            "ocr"
        ] = ocr_result

        ocr_text = (
            ocr_result.get(
                "text",
                "",
            )
            if isinstance(
                ocr_result,
                dict,
            )
            else ""
        )

        # Expose directly because existing chunker reads
        # image["ocr_text"].
        image[
            "ocr_text"
        ] = ocr_text

        metadata[
            "ocr"
        ] = ocr_result

        metadata[
            "ocr_text"
        ] = ocr_text

        # -----------------------------------------------------
        # Vision processing
        # -----------------------------------------------------

        vision_result = (
            self._get_vision_result(
                source_path,
                vision_cache,
            )
        )

        if vision_result[
            "status"
        ] == "success":

            evidence = vision_result[
                "evidence"
            ]

            result[
                "vision"
            ] = evidence

            result[
                "status"
            ] = "success"

            self._apply_visual_evidence(
                image,
                evidence,
            )

            metadata[
                "visual_evidence"
            ] = evidence

            metadata[
                "processing_status"
            ] = "success"

        else:

            result[
                "status"
            ] = vision_result[
                "status"
            ]

            result[
                "vision_error"
            ] = vision_result.get(
                "error"
            )

            metadata[
                "processing_status"
            ] = result[
                "status"
            ]

            # OCR content remains usable even if Qwen is down.
            if ocr_text:

                result[
                    "status"
                ] = "ocr_only"

                metadata[
                    "processing_status"
                ] = "ocr_only"

        return result

    # =========================================================
    # CHART PROCESSING
    # =========================================================

    def _process_chart(
        self,
        chart: dict[str, Any],
        *,
        vision_cache: dict[
            str,
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        Process chart/graph visual content with Qwen2.5-VL.

        For PDF chart candidates, image_path normally exists.

        For native XLSX charts, the Phase 5 parser currently
        extracts chart metadata but may not render a physical
        chart image. Such charts are marked rendering_required.
        """

        if not isinstance(
            chart,
            dict,
        ):
            return {
                "asset_type": "chart",
                "status": "invalid_chart",
            }

        chart_id = (
            chart.get("chart_id")
            or chart.get("id")
        )

        source_path = (
            chart.get("image_path")
            or chart.get("path")
            or chart.get(
                "source_location"
            )
        )

        metadata = chart.get(
            "metadata"
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

            chart[
                "metadata"
            ] = metadata

        result: dict[str, Any] = {
            "asset_id": chart_id,
            "asset_type": "chart",
            "source_path": source_path,
            "page_number": chart.get(
                "page_number"
            ),
            "sheet_name": chart.get(
                "sheet_name"
            ),
            "chart_type": chart.get(
                "chart_type"
            ),
            "title": chart.get(
                "title"
            ),
            "vision": None,
        }

        # -----------------------------------------------------
        # Native chart exists but no rendered image exists.
        # -----------------------------------------------------

        if (
            not source_path
            or not Path(
                source_path
            ).exists()
        ):

            result[
                "status"
            ] = "rendering_required"

            metadata[
                "processing_status"
            ] = "rendering_required"

            return result

        # -----------------------------------------------------
        # Run/reuse Qwen result.
        # -----------------------------------------------------

        vision_result = (
            self._get_vision_result(
                source_path,
                vision_cache,
            )
        )

        if vision_result[
            "status"
        ] == "success":

            evidence = vision_result[
                "evidence"
            ]

            result[
                "vision"
            ] = evidence

            result[
                "status"
            ] = "success"

            self._apply_visual_evidence(
                chart,
                evidence,
            )

            metadata[
                "visual_evidence"
            ] = evidence

            metadata[
                "processing_status"
            ] = "success"

        else:

            result[
                "status"
            ] = vision_result[
                "status"
            ]

            result[
                "vision_error"
            ] = vision_result.get(
                "error"
            )

            metadata[
                "processing_status"
            ] = result[
                "status"
            ]

        return result

    # =========================================================
    # VISION EXECUTION + CACHE
    # =========================================================

    def _get_vision_result(
        self,
        source_path: str,
        cache: dict[
            str,
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        Execute Qwen2.5-VL once per physical asset.
        """

        normalized_path = str(
            Path(
                source_path
            ).resolve()
        )

        if normalized_path in cache:
            return cache[
                normalized_path
            ]

        try:

            visual = (
                self.vision.analyze_image(
                    source_path
                )
            )

            evidence = (
                visual.to_dict()
                if hasattr(
                    visual,
                    "to_dict",
                )
                else dict(
                    visual
                )
            )

            result = {
                "status": "success",
                "evidence": evidence,
            }

        except VisionModelError as exc:

            result = {
                "status": "vision_unavailable",
                "evidence": None,
                "error": str(exc),
            }

        except Exception as exc:

            # Multimodal processing must degrade gracefully.
            result = {
                "status": "vision_error",
                "evidence": None,
                "error": str(exc),
            }

        cache[
            normalized_path
        ] = result

        return result

    # =========================================================
    # VISUAL EVIDENCE -> DOCUMENT BLOCK
    # =========================================================

    @staticmethod
    def _apply_visual_evidence(
        block: dict[str, Any],
        evidence: dict[str, Any],
    ) -> None:
        """
        Copy retrieval-relevant Qwen evidence onto the block.

        This is important because ModalityAwareChunker reads
        these top-level fields directly.
        """

        if not evidence:
            return

        description = evidence.get(
            "description"
        )

        if description:

            block[
                "visual_description"
            ] = str(
                description
            )

            # Existing chart chunker also accepts "description".
            block[
                "description"
            ] = str(
                description
            )

        visual_type = evidence.get(
            "visual_type"
        )

        if visual_type:

            block[
                "visual_type"
            ] = str(
                visual_type
            )

        # -----------------------------------------------------
        # Structured chart/image details.
        # -----------------------------------------------------

        for key in (
            "text",
            "labels",
            "axes",
            "trends",
            "numerical_relationships",
            "comparisons",
        ):

            value = evidence.get(
                key
            )

            if value is not None:

                block[
                    key
                ] = value

        if evidence.get(
            "confidence"
        ) is not None:

            block[
                "vision_confidence"
            ] = evidence[
                "confidence"
            ]

        if evidence.get(
            "model"
        ):

            block[
                "vision_model"
            ] = evidence[
                "model"
            ]

    # =========================================================
    # TABLE TEXT
    # =========================================================

    @staticmethod
    def _table_to_text(
        *,
        headers: list[Any],
        rows: list[Any],
    ) -> str:
        """
        Convert structured table data into deterministic text.

        This does not flatten or replace the structured data.
        It only provides an additional semantic representation.
        """

        lines: list[str] = []

        if headers:

            lines.append(
                " | ".join(
                    str(value)
                    for value in headers
                )
            )

        for row in rows:

            if isinstance(
                row,
                dict,
            ):

                values = [
                    str(value)
                    for value
                    in row.values()
                ]

            elif isinstance(
                row,
                (list, tuple),
            ):

                values = [
                    str(value)
                    for value
                    in row
                ]

            else:

                values = [
                    str(row)
                ]

            lines.append(
                " | ".join(
                    values
                )
            )

        return "\n".join(
            lines
        ).strip()