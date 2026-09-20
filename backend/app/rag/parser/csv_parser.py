"""CSV parser using Python's standard library."""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

from app.rag.parser.exceptions import DocumentExtractionError
from app.rag.parser.models import DocumentMetadata, ParsedDocument, TableBlock, TextBlock


_MAX_RECORDS_PER_BLOCK = 12
_MAX_RECORD_BLOCK_CHARS = 2600
_MONTH_NAMES = {
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "may": "May",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "sept": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December",
}


def _normalize_headers(headers: Sequence[str], width: int) -> list[str]:
    result: list[str] = []
    for index in range(width):
        value = headers[index].strip() if index < len(headers) else ""
        result.append(value or f"Column {index + 1}")
    return result


def _retrieval_value(value: str, label: str = "") -> str:
    """Add deterministic aliases that improve natural-language lookup."""
    cleaned = str(value or "").strip()
    full_month = _MONTH_NAMES.get(cleaned.lower())
    if full_month and full_month.lower() != cleaned.lower():
        return f"{cleaned} ({full_month})"

    # Structured files often store revenue as raw integer USD while a local LLM
    # naturally answers in millions.  Include the equivalent scale in the indexed
    # evidence so the existing strict numeric verifier can validate either form.
    if "revenue" in str(label).lower():
        numeric = cleaned.replace("$", "").replace(",", "").strip()
        try:
            amount = Decimal(numeric)
        except InvalidOperation:
            amount = None

        if amount is not None and abs(amount) >= Decimal("1000000"):
            millions = (amount / Decimal("1000000")).normalize()
            million_text = format(millions, "f").rstrip("0").rstrip(".")
            return f"{cleaned} ({million_text} million USD)"

    return cleaned


def _serialize_record(headers: Sequence[str], row: Sequence[str], row_number: int) -> str:
    width = max(len(headers), len(row))
    labels = _normalize_headers(headers, width)
    values = list(row) + [""] * max(0, width - len(row))

    fields = [
        f"{labels[index]}: {_retrieval_value(values[index], labels[index])}"
        for index in range(width)
        if str(values[index]).strip()
    ]
    return f"Row {row_number}: " + "; ".join(fields)


def _record_text_blocks(path: Path, rows: list[list[str]]) -> list[TextBlock]:
    """
    Build bounded but coherent semantic row groups.

    Three-row groups made small CSVs unnecessarily fragmented.  Questions such
    as "which month had the highest revenue" then require several retrieved
    chunks and several evidence IDs, increasing verification failures and Qwen
    retries.  Keep up to twelve rows together when the serialized text remains
    safely bounded; large CSVs still split automatically.
    """
    if len(rows) <= 1:
        return []

    headers = rows[0]
    records = rows[1:]
    serialized = [
        (index + 2, _serialize_record(headers, row, index + 2))
        for index, row in enumerate(records)
    ]
    blocks: list[TextBlock] = []
    batch: list[tuple[int, str]] = []
    batch_chars = 0

    def flush() -> None:
        nonlocal batch, batch_chars
        if not batch:
            return

        first_row = batch[0][0]
        last_row = batch[-1][0]
        row_range = f"{first_row}-{last_row}"
        lines = [
            f"CSV records from '{path.name}'.",
            "Columns: " + ", ".join(_normalize_headers(headers, len(headers))),
            f"Rows {row_range}:",
        ]
        lines.extend(text for _, text in batch)

        blocks.append(
            TextBlock(
                text="\n".join(lines),
                order=len(blocks) + 1,
                metadata={
                    "source": "csv_records",
                    "row_range": row_range,
                    "record_count": len(batch),
                },
            )
        )
        batch = []
        batch_chars = 0

    for row_number, record_text in serialized:
        projected = batch_chars + len(record_text) + (1 if batch else 0)
        if batch and (
            len(batch) >= _MAX_RECORDS_PER_BLOCK
            or projected > _MAX_RECORD_BLOCK_CHARS
        ):
            flush()

        batch.append((row_number, record_text))
        batch_chars += len(record_text) + (1 if batch_chars else 0)

    flush()
    return blocks


def parse_csv(file_path: str | Path) -> ParsedDocument:
    path = Path(file_path)
    if not path.exists():
        raise DocumentExtractionError(f"CSV not found: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(handle, dialect)
            rows = [[cell.strip() for cell in row] for row in reader]

        rows = [row for row in rows if any(cell for cell in row)]
        table = None
        if rows:
            table = TableBlock(
                rows=rows,
                headers=rows[0] if len(rows) > 1 else None,
                table_id="csv-table1",
                order=0,
                metadata={"source": "csv"},
            )

        text_blocks: list[TextBlock] = []
        if rows:
            text_blocks.append(
                TextBlock(
                    text=(
                        f"CSV table '{path.name}' contains {max(0, len(rows) - 1)} data rows "
                        f"and {len(rows[0]) if rows else 0} columns."
                    ),
                    order=0,
                    metadata={"source": "csv_summary"},
                )
            )
            text_blocks.extend(_record_text_blocks(path, rows))

        return ParsedDocument(
            metadata=DocumentMetadata(
                filename=path.name,
                file_type=".csv",
                file_size=path.stat().st_size,
                extra={
                    "row_count": max(0, len(rows) - 1),
                    "column_count": len(rows[0]) if rows else 0,
                },
            ),
            tables=[table] if table else [],
            text_blocks=text_blocks,
        )
    except DocumentExtractionError:
        raise
    except Exception as exc:
        raise DocumentExtractionError(f"Failed to parse CSV '{path.name}': {exc}") from exc
