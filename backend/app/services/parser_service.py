from __future__ import annotations

import inspect
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

from app.rag.parser.parser_factory import ParserFactory
from app.schemas.parsed_document import ParsedDocument
from app.core.logging import get_logger, log_event

logger = get_logger(__name__)


class ParserService:
    """
    Unified parser service.

    Responsibilities:
        1. Select the correct parser.
        2. Execute the existing Phase 5 parser.
        3. Support function-based and class-based parsers.
        4. Normalize parser output.
        5. Return the application's ParsedDocument schema.

    Supported Phase 5 formats:

        PDF
        DOCX
        TXT
        CSV
        XLSX
    """

    # =========================================================
    # PUBLIC API
    # =========================================================

    def parse_document(
        self,
        file_path: str | Path,
        document_id: int | None = None,
        filename: str | None = None,
    ) -> ParsedDocument:

        path = Path(file_path)
        started = time.perf_counter()

        if not path.exists():
            raise FileNotFoundError(
                f"Document file does not exist: {path}"
            )

        document_name = filename or path.name
        extension = path.suffix.lower()

        # -----------------------------------------------------
        # Select parser using existing ParserFactory.
        # -----------------------------------------------------
        parser = ParserFactory.get_parser(
            document_name
        )

        # -----------------------------------------------------
        # Execute parser.
        # -----------------------------------------------------
        raw_result = self._execute_parser(
            parser=parser,
            file_path=path,
            extension=extension,
        )

        # -----------------------------------------------------
        # Normalize parser output into application schema.
        # -----------------------------------------------------
        parsed = self._normalize_result(
            raw_result=raw_result,
            document_id=document_id,
            filename=document_name,
            extension=extension,
        )
        log_event(
            logger,
            "parsing",
            filename=document_name,
            file_type=extension,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return parsed

    # =========================================================
    # PARSER EXECUTION
    # =========================================================

    def _execute_parser(
        self,
        parser: Any,
        file_path: Path,
        extension: str,
    ) -> Dict[str, Any]:
        """
        Execute either:

            1. Function-based parser

                parse_pdf(file_path)
                parse_docx(file_path)
                parse_txt(file_path)
                parse_csv(file_path)
                parse_xlsx(file_path)

            2. Class/object-based parser

                parser.extract(file_path)

            3. Class/object-based multimodal parser

                parser.extract(
                    file_path,
                    asset_directory,
                )
        """

        # -----------------------------------------------------
        # Function-based parser
        #
        # Current ParserFactory returns functions.
        # -----------------------------------------------------
        if callable(parser) and not hasattr(
            parser,
            "extract",
        ):
            return self._execute_parser_function(
                parser=parser,
                file_path=file_path,
            )

        # -----------------------------------------------------
        # Class/object-based parser.
        # -----------------------------------------------------
        extract_method = getattr(
            parser,
            "extract",
            None,
        )

        if extract_method is not None:
            return self._execute_extract_method(
                extract_method=extract_method,
                file_path=file_path,
            )

        parser_name = getattr(
            parser,
            "__name__",
            parser.__class__.__name__,
        )

        raise TypeError(
            f"Unsupported parser implementation: "
            f"{parser_name}. "
            f"Expected either a callable parser "
            f"function or an object exposing extract()."
        )

    # =========================================================
    # FUNCTION-BASED PARSERS
    # =========================================================

    @staticmethod
    def _execute_parser_function(
        parser: Any,
        file_path: Path,
    ) -> Dict[str, Any]:
        """
        Execute a function-based parser.

        Current Phase 5 parser contract:

            parser(file_path)
        """

        if not callable(parser):
            raise TypeError(
                "Parser is not callable."
            )

        signature = inspect.signature(
            parser
        )

        parameters = list(
            signature.parameters.values()
        )

        if not parameters:
            result = parser()
        else:
            result = parser(
                str(file_path)
            )

        return ParserService._coerce_parser_result(
            result
        )

    # =========================================================
    # CLASS / OBJECT PARSERS
    # =========================================================

    @staticmethod
    def _execute_extract_method(
        extract_method: Any,
        file_path: Path,
    ) -> Dict[str, Any]:
        """
        Execute parser.extract().

        Supports:

            extract(file_path)

        and:

            extract(file_path, asset_directory)
        """

        signature = inspect.signature(
            extract_method
        )

        parameter_names = list(
            signature.parameters.keys()
        )

        if "asset_directory" in parameter_names:

            with TemporaryDirectory(
                prefix="docmindai_assets_"
            ) as asset_directory:

                result = extract_method(
                    str(file_path),
                    asset_directory,
                )

        else:
            result = extract_method(
                str(file_path)
            )

        return ParserService._coerce_parser_result(
            result
        )

    # =========================================================
    # RESULT COERCION
    # =========================================================

    @staticmethod
    def _coerce_parser_result(
        result: Any,
    ) -> Dict[str, Any]:
        """
        Convert parser output into a dictionary.

        IMPORTANT:

        Phase 5 parsers currently return:

            app.rag.parser.models.ParsedDocument

        That object provides:

            to_dict()

        We MUST preserve that structured representation.

        Otherwise CSV/DOCX/XLSX tables, images and charts
        would be converted into one string and lost.
        """

        # -----------------------------------------------------
        # No parser result.
        # -----------------------------------------------------
        if result is None:
            return {}

        # -----------------------------------------------------
        # Already a dictionary.
        # -----------------------------------------------------
        if isinstance(
            result,
            dict,
        ):
            return result

        # -----------------------------------------------------
        # Phase 5 unified ParsedDocument.
        #
        # This is the critical fix.
        # -----------------------------------------------------
        to_dict_method = getattr(
            result,
            "to_dict",
            None,
        )

        if callable(to_dict_method):

            converted = to_dict_method()

            if isinstance(
                converted,
                dict,
            ):
                return converted

            raise TypeError(
                "Parser object's to_dict() method "
                "must return a dictionary."
            )

        # -----------------------------------------------------
        # Generic dataclass fallback.
        #
        # Keeps the service compatible with future parser
        # implementations without changing current parsers.
        # -----------------------------------------------------
        if hasattr(
            result,
            "__dataclass_fields__",
        ):
            try:
                from dataclasses import asdict

                converted = asdict(result)

                if isinstance(
                    converted,
                    dict,
                ):
                    return converted

            except Exception:
                pass

        # -----------------------------------------------------
        # Final fallback.
        #
        # Only truly unstructured parser output reaches here.
        # -----------------------------------------------------
        return {
            "text": str(result)
        }

    # =========================================================
    # NORMALIZATION
    # =========================================================

    def _normalize_result(
        self,
        raw_result: Dict[str, Any],
        document_id: int | None,
        filename: str,
        extension: str,
    ) -> ParsedDocument:
        """
        Normalize raw parser output into the application's
        ParsedDocument schema.
        """

        text_blocks = self._normalize_text(
            raw_result
        )

        tables = self._normalize_list(
            raw_result.get("tables")
        )

        images = self._normalize_list(
            raw_result.get("images")
        )

        charts = self._normalize_list(
            raw_result.get("charts")
        )

        # -----------------------------------------------------
        # Parser metadata.
        # -----------------------------------------------------
        metadata = raw_result.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {
                "source_metadata": metadata
            }

        # -----------------------------------------------------
        # Unified metadata.
        # -----------------------------------------------------
        metadata.update(
            {
                "filename": filename,
                "file_type": extension,
            }
        )

        return ParsedDocument(
            document_id=document_id,
            filename=filename,
            file_type=extension,
            text_blocks=text_blocks,
            tables=tables,
            images=images,
            charts=charts,
            metadata=metadata,
            raw=raw_result,
        )

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_text(
        raw_result: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        """
        Normalize supported text structures.

        Supported source structures:

            text_blocks
            pages
            paragraphs
            text
        """

        text_blocks: list[Dict[str, Any]] = []

        # -----------------------------------------------------
        # 1. Already structured text blocks.
        # -----------------------------------------------------
        existing_blocks = raw_result.get(
            "text_blocks"
        )

        if isinstance(
            existing_blocks,
            list,
        ):

            for index, block in enumerate(
                existing_blocks
            ):

                if isinstance(
                    block,
                    dict,
                ):

                    normalized = dict(
                        block
                    )

                    normalized.setdefault(
                        "block_index",
                        index,
                    )

                    text_blocks.append(
                        normalized
                    )

                elif block:

                    text_blocks.append(
                        {
                            "block_index": index,
                            "text": str(block),
                        }
                    )

            if text_blocks:
                return text_blocks

        # -----------------------------------------------------
        # 2. Page-based output.
        # -----------------------------------------------------
        pages = raw_result.get(
            "pages"
        )

        if isinstance(
            pages,
            list,
        ):

            for page_index, page in enumerate(
                pages
            ):

                if isinstance(
                    page,
                    dict,
                ):

                    text = (
                        page.get("text")
                        or page.get("content")
                        or ""
                    )

                    if text:

                        text_blocks.append(
                            {
                                "block_index": page_index,
                                "page": page.get(
                                    "page",
                                    page_index + 1,
                                ),
                                "text": str(text),
                            }
                        )

                elif page:

                    text_blocks.append(
                        {
                            "block_index": page_index,
                            "page": page_index + 1,
                            "text": str(page),
                        }
                    )

            if text_blocks:
                return text_blocks

        # -----------------------------------------------------
        # 3. Paragraph-based output.
        # -----------------------------------------------------
        paragraphs = raw_result.get(
            "paragraphs"
        )

        if isinstance(
            paragraphs,
            list,
        ):

            for index, paragraph in enumerate(
                paragraphs
            ):

                if isinstance(
                    paragraph,
                    dict,
                ):

                    text = (
                        paragraph.get("text")
                        or paragraph.get("content")
                        or ""
                    )

                    if text:

                        normalized = dict(
                            paragraph
                        )

                        normalized.setdefault(
                            "block_index",
                            index,
                        )

                        normalized["text"] = str(
                            text
                        )

                        text_blocks.append(
                            normalized
                        )

                elif paragraph:

                    text_blocks.append(
                        {
                            "block_index": index,
                            "text": str(paragraph),
                        }
                    )

            if text_blocks:
                return text_blocks

        # -----------------------------------------------------
        # 4. Plain text.
        # -----------------------------------------------------
        text = raw_result.get(
            "text"
        )

        if text:

            text_blocks.append(
                {
                    "block_index": 0,
                    "text": str(text),
                }
            )

        return text_blocks

    # =========================================================
    # LIST NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_list(
        value: Any,
    ) -> list:

        if value is None:
            return []

        if isinstance(
            value,
            list,
        ):
            return value

        return [value]