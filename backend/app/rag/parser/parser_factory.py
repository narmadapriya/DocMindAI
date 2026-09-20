"""
DocMindAI - Phase 5
Parser Factory
----------------

Central registry for the five supported document formats:

    .pdf
    .docx
    .txt
    .csv
    .xlsx

The factory supports the existing parser implementations whether
they expose:

    1. A parser class
    2. A parser function

This is intentionally backward-compatible with the current
Phase 5 parser modules.
"""

from __future__ import annotations

import inspect
from importlib import import_module
from pathlib import Path
from typing import Any, Callable


class ParserFactory:
    """
    Factory for resolving the parser associated with a file extension.

    Supported formats:

        PDF
        DOCX
        TXT
        CSV
        XLSX
    """

    # ------------------------------------------------------------------
    # Supported parser modules
    # ------------------------------------------------------------------

    PARSER_MODULES: dict[str, str] = {
        ".pdf": "app.rag.parser.pdf_parser",
        ".docx": "app.rag.parser.docx_parser",
        ".txt": "app.rag.parser.txt_parser",
        ".csv": "app.rag.parser.csv_parser",
        ".xlsx": "app.rag.parser.excel_parser",
    }

    # ------------------------------------------------------------------
    # Explicit parser function names
    #
    # These names take priority over automatic discovery.
    # This is important because some existing parser modules are
    # function-based.
    # ------------------------------------------------------------------

    PARSER_FUNCTIONS: dict[str, tuple[str, ...]] = {
        ".pdf": (
            "parse_pdf",
            "parse_document",
        ),
        ".docx": (
            "parse_docx",
            "parse_document",
        ),
        ".txt": (
            "parse_txt",
            "parse_text",
            "parse_document",
        ),
        ".csv": (
            "parse_csv",
            "parse_document",
        ),
        ".xlsx": (
            "parse_xlsx",
            "parse_excel",
            "parse_document",
        ),
    }

    # ------------------------------------------------------------------
    # Explicit parser class names
    #
    # These are checked before generic class discovery.
    # ------------------------------------------------------------------

    PARSER_CLASSES: dict[str, tuple[str, ...]] = {
        ".pdf": (
            "PDFParser",
            "PdfParser",
        ),
        ".docx": (
            "DOCXParser",
            "DocxParser",
        ),
        ".txt": (
            "TXTParser",
            "TxtParser",
            "TextParser",
        ),
        ".csv": (
            "CSVParser",
            "CsvParser",
        ),
        ".xlsx": (
            "XLSXParser",
            "XlsxParser",
            "ExcelParser",
        ),
    }

    # ------------------------------------------------------------------
    # Extension
    # ------------------------------------------------------------------

    @classmethod
    def get_extension(cls, filename: str | Path) -> str:
        """
        Return the normalized lowercase extension.

        Examples:

            document.PDF  -> .pdf
            report.DOCX   -> .docx
            data.XLSX     -> .xlsx
        """

        extension = Path(str(filename)).suffix.lower()

        if not extension:
            raise ValueError(
                f"Could not determine file extension from: {filename}"
            )

        return extension

    # ------------------------------------------------------------------
    # Supported check
    # ------------------------------------------------------------------

    @classmethod
    def is_supported(cls, filename: str | Path) -> bool:
        """
        Return True when the filename uses one of the five
        supported Phase 5 formats.
        """

        try:
            extension = cls.get_extension(filename)
        except ValueError:
            return False

        return extension in cls.PARSER_MODULES

    # ------------------------------------------------------------------
    # Module loading
    # ------------------------------------------------------------------

    @classmethod
    def _load_module(cls, extension: str):
        """
        Import the parser module associated with an extension.
        """

        module_name = cls.PARSER_MODULES.get(extension)

        if module_name is None:
            raise ValueError(
                f"No parser module configured for extension: "
                f"{extension}"
            )

        return import_module(module_name)

    # ------------------------------------------------------------------
    # Class discovery
    # ------------------------------------------------------------------

    @classmethod
    def _find_parser_class(
        cls,
        module,
        extension: str,
    ):
        """
        Find an existing parser class inside the parser module.

        Explicit class names are preferred.

        Generic class discovery is used only as a fallback.
        """

        # --------------------------------------------------------------
        # 1. Explicit class names
        # --------------------------------------------------------------

        for class_name in cls.PARSER_CLASSES.get(
            extension,
            (),
        ):

            candidate = getattr(
                module,
                class_name,
                None,
            )

            if (
                candidate is not None
                and inspect.isclass(candidate)
                and candidate.__module__ == module.__name__
            ):
                return candidate

        # --------------------------------------------------------------
        # 2. Generic class discovery
        # --------------------------------------------------------------

        for name, candidate in vars(module).items():

            if not inspect.isclass(candidate):
                continue

            if candidate.__module__ != module.__name__:
                continue

            if "parser" not in name.lower():
                continue

            return candidate

        return None

    # ------------------------------------------------------------------
    # Function discovery
    # ------------------------------------------------------------------

    @classmethod
    def _find_parser_function(
        cls,
        module,
        extension: str,
    ) -> Callable[..., Any] | None:
        """
        Find a parser function inside the parser module.

        Explicit function names are preferred.

        This is the important compatibility layer for the current
        Phase 5 function-based parser implementations.
        """

        # --------------------------------------------------------------
        # 1. Explicit function names
        # --------------------------------------------------------------

        for function_name in cls.PARSER_FUNCTIONS.get(
            extension,
            (),
        ):

            candidate = getattr(
                module,
                function_name,
                None,
            )

            if (
                candidate is not None
                and inspect.isfunction(candidate)
                and candidate.__module__ == module.__name__
            ):
                return candidate

        # --------------------------------------------------------------
        # 2. Generic function discovery
        # --------------------------------------------------------------

        for name, candidate in vars(module).items():

            if not inspect.isfunction(candidate):
                continue

            if candidate.__module__ != module.__name__:
                continue

            lowered = name.lower()

            if (
                lowered.startswith("parse_")
                or lowered == "parse"
            ):
                return candidate

        return None

    # ------------------------------------------------------------------
    # Parser resolution
    # ------------------------------------------------------------------

    @classmethod
    def _get_parser(
        cls,
        extension: str,
    ):
        """
        Resolve either a parser class or parser function.

        Returns:

            parser class
            OR
            parser function
        """

        module = cls._load_module(extension)

        # Classes have priority.
        parser_class = cls._find_parser_class(
            module,
            extension,
        )

        if parser_class is not None:
            return parser_class

        # Existing Phase 5 modules may be function-based.
        parser_function = cls._find_parser_function(
            module,
            extension,
        )

        if parser_function is not None:
            return parser_function

        raise ImportError(
            "No parser implementation could be found in module "
            f"'{module.__name__}' for extension '{extension}'."
        )

    # ------------------------------------------------------------------
    # Backward-compatible method
    # ------------------------------------------------------------------

    @classmethod
    def _get_parser_class(
        cls,
        extension: str,
    ):
        """
        Backward-compatible parser resolver.

        Despite the historical method name, this can return either
        a parser class or a parser function.

        Existing code that calls _get_parser_class() therefore
        continues to work.
        """

        return cls._get_parser(extension)

    # ------------------------------------------------------------------
    # Public factory
    # ------------------------------------------------------------------

    @classmethod
    def get_parser(
        cls,
        filename: str | Path,
        **kwargs: Any,
    ):
        """
        Return the parser implementation for a filename.

        Examples:

            ParserFactory.get_parser("document.pdf")
            ParserFactory.get_parser("document.DOCX")

        For class-based parsers:
            returns an instantiated parser object.

        For function-based parsers:
            returns the parser callable itself.
        """

        extension = cls.get_extension(filename)

        if extension not in cls.PARSER_MODULES:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                "Supported formats are: "
                ".pdf, .docx, .txt, .csv, .xlsx"
            )

        parser = cls._get_parser(
            extension
        )

        # --------------------------------------------------------------
        # Function-based parser
        #
        # Example:
        #
        #     def parse_pdf(file_path):
        #         ...
        #
        # Do NOT call it here because file_path has not been supplied
        # to the factory yet.
        # --------------------------------------------------------------

        if inspect.isfunction(parser):
            return parser

        # --------------------------------------------------------------
        # Class-based parser
        # --------------------------------------------------------------

        if inspect.isclass(parser):

            try:
                return parser(**kwargs)

            except TypeError as exc:

                if kwargs:
                    raise TypeError(
                        f"Could not instantiate parser "
                        f"'{parser.__name__}' for extension "
                        f"'{extension}' with kwargs={kwargs}: {exc}"
                    ) from exc

                return parser()

        raise TypeError(
            f"Unsupported parser implementation type: "
            f"{type(parser).__name__}"
        )