from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


# ============================================================
# RESERVED PYTHON LOGRECORD FIELDS
# ============================================================

# Python's logging module does not allow keys such as:
#
# filename
# module
# pathname
# lineno
# message
# asctime
#
# to be supplied through ``extra={...}``.
#
# DocMindAI structured logging may legitimately want to log
# fields with those names, so they are safely renamed before
# being passed into LogRecord.

_RESERVED = (
    set(
        logging.LogRecord(
            None,
            0,
            "",
            0,
            "",
            (),
            None,
        ).__dict__
    )
    | {
        "message",
        "asctime",
    }
)


# ============================================================
# SAFE STRUCTURED FIELDS
# ============================================================

def _safe_extra_fields(
    fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert structured logging fields into keys that are safe
    for Python's logging ``extra`` argument.

    Example:

        filename="report.pdf"

    becomes:

        field_filename="report.pdf"

    This prevents:

        KeyError:
        "Attempt to overwrite 'filename' in LogRecord"

    The original value is preserved; only the structured-log
    field name changes.

    This function is intentionally centralized so callers such
    as ParserService, ingestion, retrieval, verification and
    citation services do not need special-case logging code.
    """

    safe: dict[str, Any] = {}

    for key, value in fields.items():
        safe_key = str(key)

        # Python logging also uses underscore-prefixed internal
        # fields, so avoid passing those through unchanged.
        if (
            safe_key in _RESERVED
            or safe_key.startswith("_")
        ):
            cleaned_key = (
                safe_key.lstrip("_")
                or "value"
            )

            safe_key = (
                f"field_{cleaned_key}"
            )

        # Defensive collision handling.
        #
        # Example:
        # caller supplies both:
        #
        # filename="a.pdf"
        # field_filename="b.pdf"
        #
        # No structured information should be discarded.
        while safe_key in safe:
            safe_key = (
                f"field_{safe_key}"
            )

        safe[safe_key] = value

    return safe


# ============================================================
# JSON FORMATTER
# ============================================================

class JsonFormatter(
    logging.Formatter
):
    """
    Structured JSON formatter used by DocMindAI.

    Keeps the standard logging fields plus any safe structured
    fields supplied through ``log_event``.
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        payload: dict[str, Any] = {
            "timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "level": (
                record.levelname
            ),
            "logger": (
                record.name
            ),
            "message": (
                record.getMessage()
            ),
        }

        for (
            key,
            value,
        ) in record.__dict__.items():

            if (
                key in _RESERVED
                or key.startswith("_")
            ):
                continue

            try:
                json.dumps(value)

                payload[key] = value

            except TypeError:
                payload[key] = str(
                    value
                )

        if record.exc_info:
            payload[
                "exception"
            ] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            payload,
            ensure_ascii=False,
        )


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

def configure_logging() -> None:
    """
    Configure application-wide structured JSON logging.

    LOG_LEVEL may be controlled through the existing
    environment configuration.

    Example:

        LOG_LEVEL=INFO
    """

    level_name = (
        os.getenv(
            "LOG_LEVEL",
            "INFO",
        )
        .strip()
        .upper()
    )

    level = getattr(
        logging,
        level_name,
        logging.INFO,
    )

    handler = (
        logging.StreamHandler(
            sys.stdout
        )
    )

    handler.setFormatter(
        JsonFormatter()
    )

    root = logging.getLogger()

    # Avoid duplicate handlers when FastAPI reloads.
    root.handlers.clear()

    root.addHandler(
        handler
    )

    root.setLevel(
        level
    )


# ============================================================
# LOGGER FACTORY
# ============================================================

def get_logger(
    name: str,
) -> logging.Logger:
    """
    Return a standard Python logger.

    Existing DocMindAI callers remain unchanged.
    """

    return logging.getLogger(
        name
    )


# ============================================================
# STRUCTURED EVENT LOGGER
# ============================================================

def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """
    Write one structured DocMindAI event.

    Reserved LogRecord field names are automatically renamed.

    Example:

        log_event(
            logger,
            "parsing",
            filename="report.pdf",
            latency_ms=120,
        )

    produces structured fields equivalent to:

        {
            "event": "parsing",
            "field_filename": "report.pdf",
            "latency_ms": 120
        }

    without conflicting with Python's internal
    LogRecord.filename property.
    """

    extra: dict[str, Any] = {
        "event": event,
    }

    extra.update(
        _safe_extra_fields(
            fields
        )
    )

    logger.log(
        level,
        event,
        extra=extra,
    )