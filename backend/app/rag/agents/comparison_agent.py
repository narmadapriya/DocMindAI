from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Sequence


class ComparisonAgent:
    """
    DocMindAI cross-document comparison agent.

    Preserves frozen Phase 11 behavior while supporting
    Step 10 production comparison requirements.

    Responsibilities:
        - preserve evidence per document
        - extract requested metrics conservatively
        - understand structured table/spreadsheet content
        - reject years/dates as metric values
        - safely resolve simple Excel SUM formulas
        - calculate numeric changes
        - generate citations
        - avoid unnecessary LLM calls when deterministic
          extraction already provides useful results

    IMPORTANT:
        Evidence values are preserved in their source form.

        Examples:

            Revenue: 100
                -> 100

            Revenue: 100 million
                -> 100

            Revenue: $32,500
                -> $32,500

            Revenue: 612400000
                -> 612400000

        Do not silently convert:

            100 -> $100,000,000
    """

    # =========================================================
    # INIT
    # =========================================================

    def __init__(
        self,
        reasoning_agent=None,
    ):
        self.reasoning_agent = reasoning_agent

    # =========================================================
    # COMPARE
    # =========================================================

    def compare(
        self,
        *,
        evidence: Sequence[Dict[str, Any]],
        document_ids: Sequence[str] | None = None,
        metrics: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Compare the first two selected documents using only indexed evidence.

        Phase-15 comparison quality guard:
        - no LLM call is required for numeric comparison metrics;
        - values are aligned by comparable scope before they are compared;
        - a document is never marked N/A merely because semantic top-k missed
          the relevant chunk;
        - citations are emitted only for evidence that actually supports the
          selected values.

        The API/response contract remains unchanged.
        """

        evidence = list(evidence or [])

        selected_metrics = [
            str(metric).strip()
            for metric in (
                metrics
                or ["Revenue", "Profit", "Employees"]
            )
            if str(metric).strip()
        ]

        normalized_document_ids = [
            str(document_id)
            for document_id in (document_ids or [])
        ]

        if not evidence:
            return {
                "comparison": [
                    {
                        "metric": metric,
                        "document_a": "N/A",
                        "document_b": "N/A",
                        "change": "N/A",
                    }
                    for metric in selected_metrics
                ],
                "answer": self._rows_to_markdown(
                    [
                        {
                            "metric": metric,
                            "document_a": "N/A",
                            "document_b": "N/A",
                            "change": "N/A",
                        }
                        for metric in selected_metrics
                    ],
                    evidence=[],
                    document_ids=normalized_document_ids,
                ),
                "citations": [],
                "document_count": len(
                    list(dict.fromkeys(normalized_document_ids))
                ),
            }

        aligned = self._aligned_rows(
            evidence=evidence,
            metrics=selected_metrics,
            document_ids=normalized_document_ids,
        )

        rows = aligned["rows"]
        used_evidence = aligned["used_evidence"]

        return {
            "answer": self._rows_to_markdown(
                rows,
                evidence=evidence,
                document_ids=normalized_document_ids,
            ),
            "comparison": rows,
            "citations": self._citations(used_evidence),
            "document_count": (
                len(list(dict.fromkeys(normalized_document_ids)))
                if normalized_document_ids
                else self._document_count(evidence)
            ),
        }

    # =========================================================
    # PHASE-15 FAST ALIGNED COMPARISON
    # =========================================================

    @classmethod
    def _aligned_rows(
        cls,
        *,
        evidence: Sequence[Dict[str, Any]],
        metrics: Sequence[str],
        document_ids: Sequence[str] | None,
    ) -> Dict[str, Any]:
        """Return API rows with values aligned to the same data scope."""

        ordered_ids = list(dict.fromkeys(
            str(value) for value in (document_ids or []) if str(value)
        ))

        if not ordered_ids:
            for item in evidence:
                metadata = dict(item.get("metadata") or {})
                document_id = str(
                    metadata.get("document_id")
                    or metadata.get("filename")
                    or ""
                )
                if document_id and document_id not in ordered_ids:
                    ordered_ids.append(document_id)

        first_id = ordered_ids[0] if ordered_ids else ""
        second_id = ordered_ids[1] if len(ordered_ids) > 1 else ""

        per_document: Dict[str, list[Dict[str, Any]]] = {
            first_id: [],
            second_id: [],
        }

        for item in evidence:
            metadata = dict(item.get("metadata") or {})
            document_id = str(
                metadata.get("document_id")
                or metadata.get("filename")
                or ""
            )
            if document_id in per_document:
                per_document[document_id].append(item)

        rows: list[Dict[str, str]] = []
        used_evidence: list[Dict[str, Any]] = []

        for metric in metrics:
            first_candidates = cls._metric_candidates(
                per_document.get(first_id, []),
                metric,
            )
            second_candidates = cls._metric_candidates(
                per_document.get(second_id, []),
                metric,
            )

            first_candidate, second_candidate = cls._choose_aligned_pair(
                first_candidates,
                second_candidates,
                metric=metric,
            )

            first_display = (
                cls._candidate_display(first_candidate)
                if first_candidate is not None
                else "N/A"
            )
            second_display = (
                cls._candidate_display(second_candidate)
                if second_candidate is not None
                else "N/A"
            )

            change = cls._candidate_change(
                first_candidate,
                second_candidate,
            )

            rows.append(
                {
                    "metric": str(metric),
                    "document_a": first_display,
                    "document_b": second_display,
                    "change": change,
                }
            )

            for candidate in (first_candidate, second_candidate):
                if not candidate:
                    continue
                source = candidate.get("evidence")
                if isinstance(source, dict):
                    used_evidence.append(source)

        return {
            "rows": rows,
            "used_evidence": cls._dedupe_evidence_items(used_evidence),
        }

    @classmethod
    def _metric_candidates(
        cls,
        evidence: Sequence[Dict[str, Any]],
        metric: str,
    ) -> list[Dict[str, Any]]:
        """
        Extract metric candidates from existing indexed evidence only.

        The comparison path intentionally understands the serializations
        produced by DocMindAI's own parsers:
          * "Columns: ... Row N: key: value" spreadsheet/CSV text chunks
          * DOCX/PDF flattened pipe tables
          * normal prose such as "16,070 vehicles sold"
          * PDF product catalog lines such as "FY2025 units sold: 4,820"
        """

        candidates: list[Dict[str, Any]] = []
        kind = cls._metric_kind(metric)

        for item in evidence:
            content = str(item.get("content", item.get("text", "")) or "")
            if not content.strip():
                continue

            candidates.extend(
                cls._row_serialization_candidates(
                    item=item,
                    content=content,
                    metric=metric,
                    kind=kind,
                )
            )

            candidates.extend(
                cls._flat_table_candidates_v2(
                    item=item,
                    content=content,
                    metric=metric,
                    kind=kind,
                )
            )

            candidates.extend(
                cls._prose_candidates_v2(
                    item=item,
                    content=content,
                    metric=metric,
                    kind=kind,
                )
            )

        return cls._dedupe_candidates(candidates)

    @classmethod
    def _row_serialization_candidates(
        cls,
        *,
        item: Dict[str, Any],
        content: str,
        metric: str,
        kind: str,
    ) -> list[Dict[str, Any]]:
        """Parse DocMindAI parser output of the form Row N: Key: Value; ..."""

        output: list[Dict[str, Any]] = []
        text = str(content or "")

        row_matches = list(
            re.finditer(
                r"\bRow\s+\d+\s*:\s*(.*?)(?=(?:\|\s*)?Row\s+\d+\s*:|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
        )

        for match in row_matches:
            row_text = match.group(1).strip(" |")
            fields: Dict[str, str] = {}

            for piece in re.split(r"\s*;\s*", row_text):
                pair = re.match(r"\s*([^:|]+?)\s*:\s*(.+?)\s*$", piece)
                if not pair:
                    continue
                key = cls._normalize_label(pair.group(1))
                value = pair.group(2).strip().strip("|")
                if key:
                    fields[key] = value

            metric_value = None
            metric_key = None
            for key, value in fields.items():
                if cls._field_matches_metric(key, metric, kind):
                    metric_key = key
                    metric_value = value
                    break

            if not metric_value:
                continue

            parsed = cls._parse_number_with_scale(metric_value)
            if parsed is None:
                continue

            scope, rank = cls._scope_from_fields(fields, content=text)

            output.append(
                cls._candidate(
                    parsed=parsed,
                    scope=scope,
                    rank=rank + 40,
                    evidence=item,
                    origin="row",
                    label=metric_key or metric,
                )
            )

        return output

    @classmethod
    def _flat_table_candidates_v2(
        cls,
        *,
        item: Dict[str, Any],
        content: str,
        metric: str,
        kind: str,
    ) -> list[Dict[str, Any]]:
        """Parse compact pipe tables produced for DOCX/CSV/PDF chunks."""

        metadata = dict(item.get("metadata") or {})
        if str(metadata.get("chunk_type") or "").lower() != "table":
            return []

        text = str(content or "")
        output: list[Dict[str, Any]] = []

        # Annual/full-year totals are the strongest unambiguous table scope.
        if kind == "units_sold":
            for match in re.finditer(
                r"\bFY\s*(\d{4})\s+Total\s*\|\s*([\d,]+)",
                text,
                re.IGNORECASE,
            ):
                parsed = cls._parse_number_with_scale(match.group(2))
                if parsed:
                    output.append(
                        cls._candidate(
                            parsed=parsed,
                            scope=f"annual:{match.group(1)}",
                            rank=220,
                            evidence=item,
                            origin="table-total",
                            label="FY Total",
                        )
                    )

        if kind == "revenue":
            for match in re.finditer(
                r"\bFY\s*(\d{4})\s+Total\s*\|\s*[\d,]+\s*\|\s*([$€£₹]?[\d,.]+(?:\s*(?:million|billion|thousand|[KMBTkmbt]))?)",
                text,
                re.IGNORECASE,
            ):
                parsed = cls._parse_number_with_scale(match.group(2))
                if parsed:
                    output.append(
                        cls._candidate(
                            parsed=parsed,
                            scope=f"annual:{match.group(1)}",
                            rank=220,
                            evidence=item,
                            origin="table-total",
                            label="FY Total",
                        )
                    )

        # Standard Month | Year | ... tables. This intentionally parses the
        # parser's complete table chunk rather than relying on vector rank.
        month_pattern = (
            r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        )

        if kind == "units_sold":
            pattern = re.compile(
                month_pattern
                + r"\s*\|\s*(\d{4})\s*\|\s*([\d,]+)",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                parsed = cls._parse_number_with_scale(match.group(3))
                if parsed:
                    scope = cls._month_scope(match.group(1), match.group(2))
                    output.append(
                        cls._candidate(
                            parsed=parsed,
                            scope=scope,
                            rank=120 + cls._scope_recency(scope),
                            evidence=item,
                            origin="table-month",
                            label=match.group(1),
                        )
                    )

        if kind == "revenue":
            # Monthly tables in both CSV and XLSX use the fifth field for
            # Revenue. Keep the pattern conservative to avoid matching other
            # unrelated five-column tables.
            pattern = re.compile(
                month_pattern
                + r"\s*\|\s*(\d{4})\s*\|\s*[\d,]+\s*\|\s*[$€£₹]?[\d,.]+\s*\|\s*([$€£₹]?[\d,.]+)",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                parsed = cls._parse_number_with_scale(match.group(3))
                if parsed:
                    scope = cls._month_scope(match.group(1), match.group(2))
                    output.append(
                        cls._candidate(
                            parsed=parsed,
                            scope=scope,
                            rank=120 + cls._scope_recency(scope),
                            evidence=item,
                            origin="table-month",
                            label=match.group(1),
                        )
                    )

        return output

    @classmethod
    def _prose_candidates_v2(
        cls,
        *,
        item: Dict[str, Any],
        content: str,
        metric: str,
        kind: str,
    ) -> list[Dict[str, Any]]:
        """Extract explicit grounded metric values from normal prose."""

        text = str(content or "")
        output: list[Dict[str, Any]] = []

        year_match = re.search(r"\bFY\s*(\d{4})\b|\bfiscal\s+year\s+(\d{4})\b", text, re.IGNORECASE)
        year = next((g for g in (year_match.groups() if year_match else ()) if g), None)

        if kind == "units_sold":
            # Explicit annual statements such as "16,070 vehicles sold".
            for match in re.finditer(
                r"([\d,]+)\s+(?:vehicles?|units?)\s+sold\b",
                text,
                re.IGNORECASE,
            ):
                if cls._is_year_only(match.group(1)):
                    continue
                parsed = cls._parse_number_with_scale(match.group(1))
                if not parsed:
                    continue
                annual_context = bool(
                    re.search(r"\bFY\s*\d{4}\b|\bfiscal\s+year\s+\d{4}\b|\bannual\b", text, re.IGNORECASE)
                )
                scope = f"annual:{year or 'unknown'}" if annual_context else "generic"
                rank = 210 if annual_context else 80
                output.append(
                    cls._candidate(
                        parsed=parsed,
                        scope=scope,
                        rank=rank,
                        evidence=item,
                        origin="prose",
                        label="units sold",
                    )
                )

            # PDF product catalog: sum a complete FY product breakdown only
            # when the same chunk explicitly describes a model lineup/catalog.
            values = [
                cls._parse_number_with_scale(match.group(1))
                for match in re.finditer(
                    r"FY\s*(?:\d{4}\s+)?units\s+sold\s*:\s*([\d,]+)",
                    text,
                    re.IGNORECASE,
                )
            ]
            values = [value for value in values if value is not None]
            if (
                len(values) >= 2
                and re.search(r"\b(?:model\s+lines?|product\s+catalog|lineup)\b", text, re.IGNORECASE)
            ):
                total_absolute = sum((value[2] for value in values), Decimal("0"))
                total_raw = sum((value[0] for value in values), Decimal("0"))
                output.append(
                    {
                        "raw": total_raw,
                        "scale": Decimal("1"),
                        "absolute": total_absolute,
                        "display": cls._decimal_to_source_string(total_absolute),
                        "scope": f"annual:{year or '2025'}",
                        "rank": 205,
                        "evidence": item,
                        "origin": "derived-product-total",
                        "label": "FY product units total",
                    }
                )

        elif kind == "revenue":
            # "with $612.4 million in revenue" / "revenue of $200.6 million"
            patterns = [
                r"([$€£₹]?\s*[\d,.]+(?:\s*(?:million|billion|thousand|[KMBTkmbt]))?)\s+in\s+revenue",
                r"\brevenue\s+(?:of|was|is|:)\s*([$€£₹]?\s*[\d,.]+(?:\s*(?:million|billion|thousand|[KMBTkmbt]))?)",
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    parsed = cls._parse_number_with_scale(match.group(1))
                    if not parsed:
                        continue
                    annual_context = bool(
                        re.search(r"\bFY\s*\d{4}\b|\bfiscal\s+year\s+\d{4}\b|\bannual\b", text, re.IGNORECASE)
                    )
                    scope = f"annual:{year or 'unknown'}" if annual_context else "generic"
                    output.append(
                        cls._candidate(
                            parsed=parsed,
                            scope=scope,
                            rank=205 if annual_context else 85,
                            evidence=item,
                            origin="prose",
                            label="revenue",
                        )
                    )

        # Generic explicit labels preserve Phase-11 behavior and support
        # arbitrary simple numeric metrics such as Profit and Employees.
        aliases = cls._metric_aliases(metric, kind)
        for alias in aliases:
            label_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"\b{label_pattern}\b\s*(?:[:=|-])\s*"
                r"([$€£₹]?\s*[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand|[KMBTkmbt]|%))?)",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                parsed = cls._parse_number_with_scale(match.group(1))
                if not parsed:
                    continue
                output.append(
                    cls._candidate(
                        parsed=parsed,
                        scope="generic",
                        rank=100,
                        evidence=item,
                        origin="explicit-label",
                        label=alias,
                    )
                )

        # Phase-11 plain form: "Revenue: 100 million" is covered above;
        # support the exact old "Revenue 100" variant as a lower priority.
        normalized_metric = str(metric).strip()
        if normalized_metric:
            plain = re.compile(
                rf"\b{re.escape(normalized_metric)}\b\s+"
                r"([$€£₹]?\s*[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand|[KMBTkmbt]|%))?)",
                re.IGNORECASE,
            )
            for match in plain.finditer(text):
                parsed = cls._parse_number_with_scale(match.group(1))
                if parsed and not cls._is_year_only(match.group(1)):
                    output.append(
                        cls._candidate(
                            parsed=parsed,
                            scope="generic",
                            rank=60,
                            evidence=item,
                            origin="plain-label",
                            label=metric,
                        )
                    )

        return output

    @classmethod
    def _choose_aligned_pair(
        cls,
        first: Sequence[Dict[str, Any]],
        second: Sequence[Dict[str, Any]],
        *,
        metric: str,
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
        """Choose the highest-quality pair that refers to the same scope."""

        first = list(first or [])
        second = list(second or [])

        if not first and not second:
            return None, None
        if not first:
            return None, cls._best_candidate(second)
        if not second:
            return cls._best_candidate(first), None

        first_by_scope: Dict[str, list[Dict[str, Any]]] = {}
        second_by_scope: Dict[str, list[Dict[str, Any]]] = {}

        for candidate in first:
            first_by_scope.setdefault(str(candidate.get("scope") or "generic"), []).append(candidate)
        for candidate in second:
            second_by_scope.setdefault(str(candidate.get("scope") or "generic"), []).append(candidate)

        common_scopes = set(first_by_scope).intersection(second_by_scope)
        if common_scopes:
            def scope_key(scope: str) -> tuple[int, int, str]:
                left = cls._best_candidate(first_by_scope[scope])
                right = cls._best_candidate(second_by_scope[scope])
                scope_priority = cls._scope_priority(scope)
                candidate_quality = int(left.get("rank", 0)) + int(right.get("rank", 0))
                return (scope_priority, candidate_quality, scope)

            chosen_scope = max(common_scopes, key=scope_key)
            return (
                cls._best_candidate(first_by_scope[chosen_scope]),
                cls._best_candidate(second_by_scope[chosen_scope]),
            )

        # No exact common scope. Prefer annual/full-year evidence only when
        # both sides are annual-like; otherwise use each side's strongest
        # candidate and let N/A/comparability protection prevent false math.
        first_annual = [c for c in first if str(c.get("scope", "")).startswith("annual:")]
        second_annual = [c for c in second if str(c.get("scope", "")).startswith("annual:")]
        if first_annual and second_annual:
            return cls._best_candidate(first_annual), cls._best_candidate(second_annual)

        return cls._best_candidate(first), cls._best_candidate(second)

    @staticmethod
    def _best_candidate(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
        values = list(candidates or [])
        if not values:
            return None
        return max(
            values,
            key=lambda value: (
                int(value.get("rank", 0)),
                ComparisonAgent._scope_priority(str(value.get("scope") or "generic")),
            ),
        )

    @staticmethod
    def _scope_priority(scope: str) -> int:
        value = str(scope or "generic").lower()
        if value.startswith("annual:"):
            return 500
        if value.startswith("period-total:"):
            return 450
        if value.startswith("month:"):
            return 300 + ComparisonAgent._scope_recency(value)
        if value.startswith("quarter:"):
            return 250 + ComparisonAgent._scope_recency(value)
        if value == "generic":
            return 100
        return 50

    @staticmethod
    def _scope_recency(scope: str) -> int:
        value = str(scope or "").lower()
        month_order = {
            "01": 1, "02": 2, "03": 3, "04": 4,
            "05": 5, "06": 6, "07": 7, "08": 8,
            "09": 9, "10": 10, "11": 11, "12": 12,
        }
        match = re.search(r"month:(\d{4})-(\d{2})", value)
        if match:
            return int(match.group(1)) * 12 + month_order.get(match.group(2), 0)
        match = re.search(r"quarter:(\d{4})-q([1-4])", value)
        if match:
            return int(match.group(1)) * 4 + int(match.group(2))
        return 0

    @staticmethod
    def _month_scope(month: str, year: str) -> str:
        months = {
            "jan": "01", "january": "01",
            "feb": "02", "february": "02",
            "mar": "03", "march": "03",
            "apr": "04", "april": "04",
            "may": "05",
            "jun": "06", "june": "06",
            "jul": "07", "july": "07",
            "aug": "08", "august": "08",
            "sep": "09", "sept": "09", "september": "09",
            "oct": "10", "october": "10",
            "nov": "11", "november": "11",
            "dec": "12", "december": "12",
        }
        code = months.get(str(month).strip().lower(), "00")
        return f"month:{str(year).strip()}-{code}"

    @classmethod
    def _scope_from_fields(
        cls,
        fields: Dict[str, str],
        *,
        content: str,
    ) -> tuple[str, int]:
        normalized = {cls._normalize_label(k): str(v) for k, v in fields.items()}

        month = normalized.get("month")
        year = normalized.get("year")
        if month and year and re.search(r"\d{4}", year):
            return cls._month_scope(month, re.search(r"\d{4}", year).group(0)), 100

        quarter = normalized.get("quarter")
        if quarter:
            total_match = re.search(r"FY\s*(\d{4})\s+Total", quarter, re.IGNORECASE)
            if total_match:
                return f"annual:{total_match.group(1)}", 180
            q_match = re.search(r"Q([1-4])", quarter, re.IGNORECASE)
            if q_match:
                year_match = re.search(r"\b(20\d{2})\b", content)
                year_value = year_match.group(1) if year_match else "unknown"
                return f"quarter:{year_value}-q{q_match.group(1)}", 90

        # XLSX monthly aggregate row.
        for value in normalized.values():
            if re.search(r"\bTotal\s*/\s*Avg\b", value, re.IGNORECASE):
                year_match = re.search(r"\b(20\d{2})\b", content)
                return f"period-total:{year_match.group(1) if year_match else 'unknown'}", 150

        return "generic", 20

    @staticmethod
    def _metric_kind(metric: str) -> str:
        normalized = ComparisonAgent._normalize_label(metric)
        if "revenue" in normalized:
            return "revenue"
        if (
            "unit sold" in normalized
            or "units sold" in normalized
            or normalized in {"sales units", "vehicle sales", "vehicle deliveries", "deliveries"}
        ):
            return "units_sold"
        if "employee" in normalized or "headcount" in normalized or "workforce" in normalized:
            return "employees"
        if "profit" in normalized:
            return "profit"
        if "transaction price" in normalized or "avg price" in normalized or "average price" in normalized:
            return "avg_price"
        return "generic"

    @staticmethod
    def _metric_aliases(metric: str, kind: str) -> list[str]:
        if kind == "revenue":
            return ["revenue", "revenue usd"]
        if kind == "units_sold":
            return ["units sold", "unit sales", "fy2025 units", "fy2025 units sold", "vehicle deliveries"]
        if kind == "employees":
            return ["employees", "headcount", "workforce"]
        if kind == "profit":
            return ["profit", "net profit", "operating profit"]
        if kind == "avg_price":
            return ["avg transaction price", "average transaction price", "avg price"]
        return [str(metric).strip()]

    @classmethod
    def _field_matches_metric(cls, key: str, metric: str, kind: str) -> bool:
        normalized_key = cls._normalize_label(key)
        aliases = [cls._normalize_label(value) for value in cls._metric_aliases(metric, kind)]
        return any(
            normalized_key == alias
            or normalized_key.endswith(alias)
            or alias in normalized_key
            for alias in aliases
            if alias
        )

    @classmethod
    def _parse_number_with_scale(
        cls,
        value: str,
    ) -> tuple[Decimal, Decimal, Decimal, str] | None:
        text = str(value or "").strip()
        if not text or text.upper() == "N/A":
            return None

        match = re.search(
            r"([$€£₹]?\s*[-+]?\d[\d,]*(?:\.\d+)?)\s*(million|billion|trillion|thousand|[KMBTkmbt])?",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None

        numeric_text = re.sub(r"[^0-9+\-.]", "", match.group(1))
        try:
            raw = Decimal(numeric_text)
        except InvalidOperation:
            return None

        suffix = str(match.group(2) or "").lower()
        scale = {
            "thousand": Decimal("1000"),
            "k": Decimal("1000"),
            "million": Decimal("1000000"),
            "m": Decimal("1000000"),
            "billion": Decimal("1000000000"),
            "b": Decimal("1000000000"),
            "trillion": Decimal("1000000000000"),
            "t": Decimal("1000000000000"),
        }.get(suffix, Decimal("1"))

        display = cls._decimal_to_source_string(raw)
        if suffix:
            # Preserve the legacy Phase-11 display contract: "100 million"
            # is displayed as "100", while scale is retained internally for
            # comparable arithmetic.
            display = cls._decimal_to_source_string(raw)

        return raw, scale, raw * scale, display

    @staticmethod
    def _candidate(
        *,
        parsed: tuple[Decimal, Decimal, Decimal, str],
        scope: str,
        rank: int,
        evidence: Dict[str, Any],
        origin: str,
        label: str,
    ) -> Dict[str, Any]:
        raw, scale, absolute, display = parsed
        return {
            "raw": raw,
            "scale": scale,
            "absolute": absolute,
            "display": display,
            "scope": scope,
            "rank": int(rank),
            "evidence": evidence,
            "origin": origin,
            "label": label,
        }

    @classmethod
    def _candidate_display(cls, candidate: Dict[str, Any] | None) -> str:
        if not candidate:
            return "N/A"
        return str(candidate.get("display") or "N/A")

    @classmethod
    def _candidate_change(
        cls,
        first: Dict[str, Any] | None,
        second: Dict[str, Any] | None,
    ) -> str:
        if not first or not second:
            return "N/A"

        # Never calculate across different non-generic scopes.
        first_scope = str(first.get("scope") or "generic")
        second_scope = str(second.get("scope") or "generic")
        if first_scope != second_scope and first_scope != "generic" and second_scope != "generic":
            return "N/A"

        first_scale = Decimal(str(first.get("scale", 1)))
        second_scale = Decimal(str(second.get("scale", 1)))

        if first_scale == second_scale:
            a = Decimal(str(first.get("raw", 0)))
            b = Decimal(str(second.get("raw", 0)))
        else:
            a = Decimal(str(first.get("absolute", 0)))
            b = Decimal(str(second.get("absolute", 0)))

        difference = b - a
        if difference == 0:
            return "0"

        rendered = cls._decimal_to_source_string(abs(difference))
        return ("+" if difference > 0 else "-") + rendered

    @staticmethod
    def _dedupe_candidates(candidates: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        result: list[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for candidate in candidates:
            evidence = candidate.get("evidence") or {}
            key = (
                str(candidate.get("scope") or "generic"),
                str(candidate.get("absolute")),
                str(evidence.get("chunk_id") or id(evidence)),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)
        return result

    @staticmethod
    def _dedupe_evidence_items(evidence: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        result: list[Dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(evidence):
            metadata = dict(item.get("metadata") or {})
            key = str(
                item.get("chunk_id")
                or item.get("postgres_chunk_id")
                or f"{metadata.get('document_id','')}:{metadata.get('page_number','')}:{metadata.get('sheet_name','')}:{index}"
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @staticmethod
    def _rows_to_markdown(
        rows: Sequence[Dict[str, str]],
        *,
        evidence: Sequence[Dict[str, Any]],
        document_ids: Sequence[str] | None,
    ) -> str:
        labels: Dict[str, str] = {}
        for item in evidence:
            metadata = dict(item.get("metadata") or {})
            document_id = str(metadata.get("document_id") or "")
            filename = str(metadata.get("filename") or document_id or "")
            if document_id and filename:
                labels[document_id] = filename

        ids = list(dict.fromkeys(str(value) for value in (document_ids or []) if str(value)))
        first_label = labels.get(ids[0], ids[0]) if ids else "Document A"
        second_label = labels.get(ids[1], ids[1]) if len(ids) > 1 else "Document B"

        lines = [
            f"| Metric | {first_label} | {second_label} | Change |",
            "|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                "| {metric} | {document_a} | {document_b} | {change} |".format(
                    **row
                )
            )
        return "\n".join(lines)

    # =========================================================
    # PROMPT
    # =========================================================

    @staticmethod
    def _build_prompt(
        *,
        evidence: Sequence[Dict[str, Any]],
        metrics: Sequence[str],
        document_ids: Sequence[str] | None = None,
    ) -> str:

        metric_text = ", ".join(
            metrics
        )

        document_text = ", ".join(
            str(document_id)

            for document_id
            in (
                document_ids
                or []
            )
        )

        return f"""
You are the cross-document comparison stage of DocMindAI.

Compare ONLY the supplied documents and ONLY use supplied
evidence.

Documents:
{document_text}

Metrics:
{metric_text}

Rules:

1. Never invent a value.
2. Keep each value associated with its source document.
3. Search text, tables, spreadsheet rows, OCR and charts.
4. Do not treat a year such as 2024 or 2025 as Revenue,
   Profit, Employees or another requested metric.
5. Prefer explicit totals when totals and individual rows
   are both available.
6. For annual metrics prefer:
       FY2025 Total
       Annual Total
       Full Year
       Total
   over monthly or quarterly values.
7. If a metric is unavailable in a document, return N/A.
8. Do not invent an aggregate from qualitative statements.
9. Change = Document B - Document A only when both values
   are numeric and comparable.
10. Return ONLY a Markdown table.

Required format:

| Metric | Document A | Document B | Change |
|---|---|---|---|
| Revenue | ... | ... | ... |

Evidence:

{ComparisonAgent._format_evidence(evidence)}
""".strip()

    # =========================================================
    # FORMAT EVIDENCE
    # =========================================================

    @staticmethod
    def _format_evidence(
        evidence: Sequence[Dict[str, Any]],
    ) -> str:

        blocks: list[str] = []

        for index, item in enumerate(
            evidence,
            start=1,
        ):
            metadata = dict(
                item.get(
                    "metadata"
                )
                or {}
            )

            blocks.append(
                "\n".join(
                    [
                        (
                            f"EVIDENCE "
                            f"{index}"
                        ),
                        (
                            "DOCUMENT_ID: "
                            + str(
                                metadata.get(
                                    "document_id",
                                    "",
                                )
                            )
                        ),
                        (
                            "DOCUMENT: "
                            + str(
                                metadata.get(
                                    "filename",
                                    "Unknown",
                                )
                            )
                        ),
                        (
                            "PAGE: "
                            + str(
                                metadata.get(
                                    "page_number",
                                    "",
                                )
                            )
                        ),
                        (
                            "SHEET: "
                            + str(
                                metadata.get(
                                    "sheet_name",
                                    "",
                                )
                            )
                        ),
                        (
                            "TYPE: "
                            + str(
                                metadata.get(
                                    "chunk_type",
                                    "text",
                                )
                            )
                        ),
                        (
                            "CONTENT:\n"
                            + str(
                                item.get(
                                    "content",
                                    item.get(
                                        "text",
                                        "",
                                    ),
                                )
                            )
                        ),
                    ]
                )
            )

        return "\n\n".join(
            blocks
        )

    # =========================================================
    # REASONING RESULT
    # =========================================================

    @staticmethod
    def _reasoning_answer(
        result: Any,
    ) -> str:

        if isinstance(
            result,
            dict,
        ):
            return str(
                result.get(
                    "answer"
                )
                or result.get(
                    "content"
                )
                or result.get(
                    "response"
                )
                or ""
            )

        return str(
            result
            or ""
        )

    # =========================================================
    # FALLBACK COMPARISON
    # =========================================================

    @staticmethod
    def _fallback_comparison(
        *,
        evidence: Sequence[Dict[str, Any]],
        metrics: Sequence[str],
        document_ids: Sequence[str] | None = None,
    ) -> str:

        documents = (
            ComparisonAgent
            ._group_by_document(
                evidence=evidence,
                document_ids=document_ids,
            )
        )

        names = list(
            documents.keys()
        )

        first = (
            names[0]
            if len(names) > 0
            else "Document A"
        )

        second = (
            names[1]
            if len(names) > 1
            else "Document B"
        )

        first_data = (
            documents.get(
                first,
                {},
            )
        )

        second_data = (
            documents.get(
                second,
                {},
            )
        )

        first_label = str(
            first_data.get(
                "filename",
                first,
            )
        )

        second_label = str(
            second_data.get(
                "filename",
                second,
            )
        )

        lines = [
            (
                "| Metric | "
                f"{first_label} | "
                f"{second_label} | "
                "Change |"
            ),
            "|---|---|---|---|",
        ]

        for metric in metrics:

            first_value = (
                ComparisonAgent
                ._find_metric(
                    first_data.get(
                        "contents",
                        [],
                    ),
                    metric,
                )
            )

            second_value = (
                ComparisonAgent
                ._find_metric(
                    second_data.get(
                        "contents",
                        [],
                    ),
                    metric,
                )
            )

            change = (
                ComparisonAgent
                ._numeric_change(
                    first_value,
                    second_value,
                )
            )

            lines.append(
                f"| {metric} | "
                f"{first_value} | "
                f"{second_value} | "
                f"{change} |"
            )

        return "\n".join(
            lines
        )

    # =========================================================
    # GROUP BY DOCUMENT
    # =========================================================

    @staticmethod
    def _group_by_document(
        *,
        evidence: Sequence[Dict[str, Any]],
        document_ids: Sequence[str] | None = None,
    ) -> Dict[str, Dict[str, Any]]:

        grouped: Dict[
            str,
            Dict[str, Any],
        ] = {}

        # -----------------------------------------------------
        # Preserve requested document ordering.
        # -----------------------------------------------------

        for document_id in (
            document_ids
            or []
        ):
            grouped[
                str(
                    document_id
                )
            ] = {
                "filename":
                    str(
                        document_id
                    ),
                "contents":
                    [],
            }

        for item in evidence:

            metadata = dict(
                item.get(
                    "metadata"
                )
                or {}
            )

            document_id = str(
                metadata.get(
                    "document_id"
                )
                or metadata.get(
                    "filename"
                )
                or "Unknown"
            )

            filename = str(
                metadata.get(
                    "filename"
                )
                or document_id
            )

            content = str(
                item.get(
                    "content",
                    item.get(
                        "text",
                        "",
                    ),
                )
            )

            if (
                document_id
                not in grouped
            ):
                grouped[
                    document_id
                ] = {
                    "filename":
                        filename,
                    "contents":
                        [],
                }

            grouped[
                document_id
            ][
                "filename"
            ] = filename

            grouped[
                document_id
            ][
                "contents"
            ].append(
                content
            )

        return grouped

    # =========================================================
    # FIND METRIC
    # =========================================================

    @classmethod
    def _find_metric(
        cls,
        contents: Sequence[str],
        metric: str,
    ) -> str:
        """
        Find a metric conservatively.

        Priority:

            1. Structured table/spreadsheet extraction
            2. Explicit labelled text value
            3. N/A
        """

        metric = str(
            metric
            or ""
        ).strip()

        if not metric:
            return "N/A"

        # =====================================================
        # STRUCTURED TABLES
        # =====================================================

        table_candidates: list[
            tuple[int, str]
        ] = []

        for content in contents:
            table_candidates.extend(
                cls._table_metric_candidates(
                    content=content,
                    metric=metric,
                )
            )

        if table_candidates:
            table_candidates.sort(
                key=lambda item:
                    item[0],
                reverse=True,
            )

            return (
                table_candidates[
                    0
                ][1]
            )

        # =====================================================
        # EXPLICIT LABELLED VALUES
        # =====================================================

        explicit_candidates: list[
            tuple[int, str]
        ] = []

        for content in contents:
            explicit_candidates.extend(
                cls._explicit_metric_candidates(
                    content=content,
                    metric=metric,
                )
            )

        if explicit_candidates:
            explicit_candidates.sort(
                key=lambda item:
                    item[0],
                reverse=True,
            )

            return (
                explicit_candidates[
                    0
                ][1]
            )

        return "N/A"

    # =========================================================
    # STRUCTURED TABLE CANDIDATES
    # =========================================================

    @classmethod
    def _table_metric_candidates(
        cls,
        *,
        content: str,
        metric: str,
    ) -> list[tuple[int, str]]:

        text = str(
            content
            or ""
        )

        if not text.strip():
            return []

        candidates: list[
            tuple[int, str]
        ] = []

        for delimiter in (
            "|",
            "\t",
            ",",
        ):

            rows = (
                cls._split_table_rows(
                    text=text,
                    delimiter=delimiter,
                )
            )

            if len(rows) < 2:
                continue

            # -------------------------------------------------
            # Search each row for a metric header.
            # -------------------------------------------------

            for (
                header_index,
                header,
            ) in enumerate(
                rows
            ):

                metric_column = (
                    cls._find_metric_column(
                        cells=header,
                        metric=metric,
                    )
                )

                if metric_column is None:
                    continue

                # ---------------------------------------------
                # Inspect rows beneath that header.
                # ---------------------------------------------

                for (
                    relative_index,
                    row,
                ) in enumerate(
                    rows[
                        header_index
                        + 1:
                    ]
                ):

                    if (
                        metric_column
                        >= len(row)
                    ):
                        continue

                    raw_value = str(
                        row[
                            metric_column
                        ]
                    ).strip()

                    if not raw_value:
                        continue

                    # =========================================
                    # SIMPLE EXCEL FORMULA RESOLUTION
                    # =========================================

                    if raw_value.startswith(
                        "="
                    ):
                        raw_value = (
                            cls._resolve_formula_value(
                                raw_value=(
                                    raw_value
                                ),
                                rows=rows,
                                row_index=(
                                    header_index
                                    + 1
                                    + relative_index
                                ),
                                column_index=(
                                    metric_column
                                ),
                            )
                        )

                    if (
                        not raw_value
                        or raw_value == "N/A"
                    ):
                        continue

                    if not cls._looks_numeric(
                        raw_value
                    ):
                        continue

                    if cls._is_year_only(
                        raw_value
                    ):
                        continue

                    # -----------------------------------------
                    # Everything before the metric column is
                    # used as the row label.
                    # -----------------------------------------

                    label = " ".join(
                        str(cell)

                        for cell
                        in row[
                            :metric_column
                        ]
                    ).lower()

                    score = 40

                    # =========================================
                    # TOTAL PRIORITY
                    # =========================================

                    if (
                        "fy2025 total"
                        in label
                    ):
                        score += 100

                    elif (
                        "full year"
                        in label
                        or "full-year"
                        in label
                        or "annual total"
                        in label
                    ):
                        score += 95

                    elif (
                        "total / avg"
                        in label
                    ):
                        score += 80

                    elif re.search(
                        r"\btotal\b",
                        label,
                    ):
                        score += 70

                    elif re.search(
                        r"\bq4\b",
                        label,
                    ):
                        score += 20

                    # Slight preference for lower/final rows.
                    score += min(
                        relative_index,
                        10,
                    )

                    formatted = (
                        cls._format_metric_value(
                            raw_value,
                            header=(
                                header[
                                    metric_column
                                ]
                            ),
                            metric=metric,
                        )
                    )

                    if formatted != "N/A":
                        candidates.append(
                            (
                                score,
                                formatted,
                            )
                        )

        return candidates

    # =========================================================
    # SAFE EXCEL SUM FORMULA RESOLUTION
    # =========================================================

    @classmethod
    def _resolve_formula_value(
        cls,
        *,
        raw_value: str,
        rows: Sequence[Sequence[str]],
        row_index: int,
        column_index: int,
    ) -> str:
        """
        Safely resolve a simple same-column Excel SUM(...) formula
        from already-retrieved table evidence.

        A retrieved chunk may contain multiple logical tables, so
        aggregation is restricted to the nearest preceding period
        table header (Quarter or Month). Arbitrary Excel formulas
        are never executed.
        """

        formula = str(raw_value or "").strip()

        formula_match = re.fullmatch(
            r"=SUM\(([A-Z]+)(\d+):([A-Z]+)(\d+)\)",
            formula,
            re.IGNORECASE,
        )

        if not formula_match:
            return "N/A"

        start_column = formula_match.group(1).upper()
        end_column = formula_match.group(3).upper()

        if start_column != end_column:
            return "N/A"

        quarter_pattern = re.compile(r"^q[1-4]$", re.IGNORECASE)
        month_pattern = re.compile(
            r"^(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)$",
            re.IGNORECASE,
        )

        table_start_index: int | None = None
        period_type: str | None = None

        for index in range(row_index - 1, -1, -1):
            row = rows[index]
            if not row:
                continue

            normalized_cells = [
                cls._normalize_label(str(cell))
                for cell in row
            ]

            if any(cell == "quarter" for cell in normalized_cells):
                table_start_index = index + 1
                period_type = "quarter"
                break

            if any(cell == "month" for cell in normalized_cells):
                table_start_index = index + 1
                period_type = "month"
                break

        if table_start_index is None or period_type is None:
            return "N/A"

        values: list[Decimal] = []

        for row in rows[table_start_index:row_index]:
            if not row or column_index >= len(row):
                continue

            label_cells = [
                str(cell).strip()
                for cell in row[:column_index]
                if str(cell).strip()
            ]

            if not label_cells:
                continue

            valid_period_row = any(
                (
                    period_type == "quarter"
                    and quarter_pattern.fullmatch(label)
                )
                or (
                    period_type == "month"
                    and month_pattern.fullmatch(label)
                )
                for label in label_cells
            )

            if not valid_period_row:
                continue

            raw_cell = str(row[column_index]).strip()

            if not raw_cell or raw_cell.startswith("="):
                continue

            if cls._is_year_only(raw_cell):
                continue

            parsed_value = cls._parse_decimal(raw_cell)
            if parsed_value is not None:
                values.append(parsed_value)

        if not values:
            return "N/A"

        total = sum(values, Decimal("0"))
        return cls._decimal_to_source_string(total)

    # =========================================================
    # SPLIT TABLE
    # =========================================================

    @staticmethod
    def _split_table_rows(
        *,
        text: str,
        delimiter: str,
    ) -> list[list[str]]:

        rows: list[
            list[str]
        ] = []

        for raw_line in str(
            text
            or ""
        ).splitlines():

            line = (
                raw_line.strip()
            )

            if not line:
                continue

            if delimiter not in line:
                continue

            if delimiter == "|":
                line = (
                    line.strip(
                        "|"
                    )
                )

            cells = [
                cell.strip()

                for cell
                in line.split(
                    delimiter
                )
            ]

            if len(cells) >= 2:
                rows.append(
                    cells
                )

        return rows

    # =========================================================
    # FIND METRIC COLUMN
    # =========================================================

    @staticmethod
    def _find_metric_column(
        *,
        cells: Sequence[str],
        metric: str,
    ) -> int | None:

        normalized_metric = (
            ComparisonAgent
            ._normalize_label(
                metric
            )
        )

        if not normalized_metric:
            return None

        for index, cell in enumerate(
            cells
        ):

            normalized_cell = (
                ComparisonAgent
                ._normalize_label(
                    cell
                )
            )

            if (
                normalized_cell
                == normalized_metric
            ):
                return index

            if (
                normalized_metric
                in normalized_cell
            ):
                return index

        return None

    # =========================================================
    # EXPLICIT METRIC CANDIDATES
    # =========================================================

    @classmethod
    def _explicit_metric_candidates(
        cls,
        *,
        content: str,
        metric: str,
    ) -> list[tuple[int, str]]:
        """
        Extract text-style metric values while preserving
        frozen Phase 11 behavior.

        Examples:

            Revenue: 100
                -> 100

            Revenue: 100 million
                -> 100

            Revenue: $32,500
                -> $32,500

            Employees: 850
                -> 850

        Magnitude suffixes are consumed but not included in
        the displayed value.
        """

        text = str(
            content
            or ""
        )

        if not text.strip():
            return []

        candidates: list[
            tuple[int, str]
        ] = []

        metric_pattern = (
            re.escape(
                metric
            )
        )

        # -----------------------------------------------------
        # Capture numeric value only.
        # -----------------------------------------------------

        value_pattern = (
            r"("
            r"[$€£₹]?\s*"
            r"[-+]?"
            r"\d[\d,]*"
            r"(?:\.\d+)?"
            r")"
        )

        # -----------------------------------------------------
        # Consume optional units without placing them in the
        # returned capture group.
        # -----------------------------------------------------

        unit_suffix = (
            r"(?:"
            r"\s*"
            r"(?:"
            r"thousand|"
            r"million|"
            r"billion|"
            r"trillion|"
            r"%|"
            r"[KMBTkmbt]"
            r")"
            r"\b"
            r")?"
        )

        patterns = [
            # -------------------------------------------------
            # Annual/total label.
            # -------------------------------------------------
            (
                120,
                re.compile(
                    (
                        rf"\b(?:"
                        rf"FY\s*\d{{4}}\s+Total|"
                        rf"Annual\s+Total|"
                        rf"Full[- ]Year|"
                        rf"Total"
                        rf")\s+"
                        rf"{metric_pattern}"
                        rf"\s*"
                        rf"(?:[:=|\-])"
                        rf"\s*"
                        + value_pattern
                        + unit_suffix
                    ),
                    re.IGNORECASE,
                ),
            ),

            # -------------------------------------------------
            # Phase 11:
            #
            # Revenue: 100
            # Revenue: 100 million
            # Revenue = 100
            # Revenue | 100
            # -------------------------------------------------
            (
                100,
                re.compile(
                    (
                        rf"\b{metric_pattern}\b"
                        rf"\s*"
                        rf"(?:[:=|])"
                        rf"\s*"
                        + value_pattern
                        + unit_suffix
                    ),
                    re.IGNORECASE,
                ),
            ),

            # Revenue - 100
            (
                90,
                re.compile(
                    (
                        rf"\b{metric_pattern}\b"
                        rf"\s*-\s*"
                        + value_pattern
                        + unit_suffix
                    ),
                    re.IGNORECASE,
                ),
            ),

            # Revenue,100
            (
                85,
                re.compile(
                    (
                        rf"\b{metric_pattern}\b"
                        rf"\s*,\s*"
                        + value_pattern
                        + unit_suffix
                    ),
                    re.IGNORECASE,
                ),
            ),

            # -------------------------------------------------
            # Legacy plain form:
            #
            # Revenue 100
            #
            # Year/date protection below prevents 2025 from
            # being accepted as Revenue.
            # -------------------------------------------------
            (
                60,
                re.compile(
                    (
                        rf"\b{metric_pattern}\b"
                        rf"\s+"
                        + value_pattern
                        + unit_suffix
                    ),
                    re.IGNORECASE,
                ),
            ),
        ]

        for (
            base_score,
            pattern,
        ) in patterns:

            for match in pattern.finditer(
                text
            ):

                raw_value = str(
                    match.group(1)
                ).strip()

                if not raw_value:
                    continue

                if cls._is_year_only(
                    raw_value
                ):
                    continue

                context_start = max(
                    0,
                    match.start()
                    - 50,
                )

                context_end = min(
                    len(text),
                    match.end()
                    + 50,
                )

                context = (
                    text[
                        context_start:
                        context_end
                    ].lower()
                )

                if cls._looks_like_date_context(
                    context=context,
                    value=raw_value,
                ):
                    continue

                formatted = (
                    cls._format_metric_value(
                        raw_value,
                        header=metric,
                        metric=metric,
                    )
                )

                if formatted == "N/A":
                    continue

                score = (
                    base_score
                )

                if (
                    "fy2025 total"
                    in context
                    or "fy 2025 total"
                    in context
                ):
                    score += 50

                elif (
                    "annual total"
                    in context
                    or "full year"
                    in context
                    or "full-year"
                    in context
                ):
                    score += 45

                elif (
                    "total"
                    in context
                ):
                    score += 30

                candidates.append(
                    (
                        score,
                        formatted,
                    )
                )

        return candidates

    # =========================================================
    # YEAR PROTECTION
    # =========================================================

    @staticmethod
    def _is_year_only(
        value: str,
    ) -> bool:
        """
        Prevent years such as 2024/2025 from becoming metrics.
        """

        cleaned = (
            str(
                value
                or ""
            )
            .replace(
                ",",
                "",
            )
            .replace(
                "$",
                "",
            )
            .replace(
                "€",
                "",
            )
            .replace(
                "£",
                "",
            )
            .replace(
                "₹",
                "",
            )
            .strip()
        )

        if not re.fullmatch(
            r"\d{4}",
            cleaned,
        ):
            return False

        try:
            number = int(
                cleaned
            )

        except ValueError:
            return False

        return (
            1900
            <= number
            <= 2100
        )

    # =========================================================
    # DATE CONTEXT PROTECTION
    # =========================================================

    @classmethod
    def _looks_like_date_context(
        cls,
        *,
        context: str,
        value: str,
    ) -> bool:

        if not cls._is_year_only(
            value
        ):
            return False

        date_words = (
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
            "month",
            "monthly",
            "quarter",
            "quarterly",
            "trend",
            "fy",
        )

        lowered = str(
            context
            or ""
        ).lower()

        return any(
            word in lowered

            for word
            in date_words
        )

    # =========================================================
    # NUMERIC CHECK
    # =========================================================

    @staticmethod
    def _looks_numeric(
        value: str,
    ) -> bool:

        return (
            re.search(
                r"[-+]?\d",
                str(
                    value
                    or ""
                ),
            )
            is not None
        )

    # =========================================================
    # FORMAT METRIC VALUE
    # =========================================================

    @classmethod
    def _format_metric_value(
        cls,
        value: str,
        *,
        header: str,
        metric: str,
    ) -> str:
        """
        Preserve evidence representation exactly.

        Phase 11 compatibility:

            100 -> 100

        not:

            100 -> $100,000,000
        """

        cleaned = re.sub(
            r"\s+",
            " ",
            str(
                value
                or ""
            ),
        ).strip()

        if not cleaned:
            return "N/A"

        if cls._is_year_only(
            cleaned
        ):
            return "N/A"

        return cleaned

    # =========================================================
    # NUMERIC CHANGE
    # =========================================================

    @staticmethod
    def _numeric_change(
        first: str,
        second: str,
    ) -> str:
        """
        Calculate:

            Document B - Document A

        Frozen Phase 11 formatting:

            100 -> 120
                +20

            120 -> 100
                -20

            100 -> 100
                0

            100.5 -> 120.75
                +20.25
        """

        if (
            first == "N/A"
            or second == "N/A"
        ):
            return "N/A"

        try:
            a = float(
                ComparisonAgent
                ._clean_numeric_for_legacy(
                    first
                )
            )

            b = float(
                ComparisonAgent
                ._clean_numeric_for_legacy(
                    second
                )
            )

            difference = round(
                b - a,
                2,
            )

        except (
            ValueError,
            TypeError,
            AttributeError,
        ):
            return "N/A"

        # Avoid negative zero.
        if difference == 0:
            return "0"

        # Remove unnecessary .0.
        if float(
            difference
        ).is_integer():
            value = str(
                int(
                    difference
                )
            )

        else:
            value = (
                f"{difference:.2f}"
                .rstrip("0")
                .rstrip(".")
            )

        if difference > 0:
            return (
                "+"
                + value
            )

        return value

    # =========================================================
    # LEGACY NUMERIC CLEANING
    # =========================================================

    @staticmethod
    def _clean_numeric_for_legacy(
        value: str,
    ) -> str:

        text = str(
            value
            or ""
        ).strip()

        # -----------------------------------------------------
        # Avoid calculations where a textual magnitude remains
        # because those values may not be comparable directly.
        # -----------------------------------------------------

        if re.search(
            r"\b("
            r"million|"
            r"billion|"
            r"trillion|"
            r"thousand"
            r")\b",
            text,
            re.IGNORECASE,
        ):
            raise ValueError(
                "Unit-bearing value."
            )

        text = (
            text
            .replace(
                ",",
                "",
            )
            .replace(
                "$",
                "",
            )
            .replace(
                "€",
                "",
            )
            .replace(
                "£",
                "",
            )
            .replace(
                "₹",
                "",
            )
            .replace(
                "%",
                "",
            )
            .strip()
        )

        return text

    # =========================================================
    # DECIMAL PARSER
    # =========================================================

    @staticmethod
    def _parse_decimal(
        value: str,
    ) -> Decimal | None:

        text = str(
            value
            or ""
        ).strip()

        if (
            not text
            or text.upper() == "N/A"
        ):
            return None

        text = (
            text
            .replace(
                ",",
                "",
            )
            .replace(
                "$",
                "",
            )
            .replace(
                "€",
                "",
            )
            .replace(
                "£",
                "",
            )
            .replace(
                "₹",
                "",
            )
            .replace(
                "%",
                "",
            )
            .strip()
        )

        match = re.search(
            r"[-+]?"
            r"\d+(?:\.\d+)?",
            text,
        )

        if not match:
            return None

        try:
            return Decimal(
                match.group(0)
            )

        except InvalidOperation:
            return None

    # =========================================================
    # DECIMAL TO SOURCE STRING
    # =========================================================

    @staticmethod
    def _decimal_to_source_string(
        value: Decimal,
    ) -> str:

        if (
            value
            == value.to_integral_value()
        ):
            return str(
                int(
                    value
                )
            )

        normalized = (
            value.normalize()
        )

        return format(
            normalized,
            "f",
        )

    # =========================================================
    # LABEL NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_label(
        value: str,
    ) -> str:

        return (
            re.sub(
                r"[^a-z0-9]+",
                " ",
                str(
                    value
                    or ""
                ).lower(),
            )
            .strip()
        )

    # =========================================================
    # RESULT PARSING
    # =========================================================

    @staticmethod
    def _extract_rows(
        answer: str,
    ) -> List[Dict[str, str]]:

        rows: list[
            Dict[str, str]
        ] = []

        for line in str(
            answer
            or ""
        ).splitlines():

            stripped = (
                line.strip()
            )

            if not stripped.startswith(
                "|"
            ):
                continue

            cells = [
                cell.strip()

                for cell
                in stripped
                .strip("|")
                .split("|")
            ]

            if len(cells) != 4:
                continue

            # Header.
            if (
                cells[0]
                .lower()
                == "metric"
            ):
                continue

            # Markdown separator.
            if all(
                re.fullmatch(
                    r":?-{2,}:?",
                    cell,
                )
                is not None

                for cell
                in cells
            ):
                continue

            if not cells[0]:
                continue

            rows.append(
                {
                    "metric":
                        cells[0],

                    "document_a":
                        cells[1],

                    "document_b":
                        cells[2],

                    "change":
                        cells[3],
                }
            )

        return rows

    # =========================================================
    # ROW QUALITY
    # =========================================================

    @staticmethod
    def _rows_are_useful(
        rows: Sequence[Dict[str, str]],
    ) -> bool:

        if not rows:
            return False

        for row in rows:

            first = str(
                row.get(
                    "document_a",
                    "",
                )
            ).strip()

            second = str(
                row.get(
                    "document_b",
                    "",
                )
            ).strip()

            if (
                first
                and first.upper()
                != "N/A"
            ):
                return True

            if (
                second
                and second.upper()
                != "N/A"
            ):
                return True

        return False

    # =========================================================
    # CITATIONS
    # =========================================================

    @staticmethod
    def _citations(
        evidence: Sequence[Dict[str, Any]],
    ) -> List[str]:

        result: list[str] = []
        seen: set[str] = set()

        for item in evidence:

            metadata = dict(
                item.get(
                    "metadata"
                )
                or {}
            )

            filename = (
                metadata.get(
                    "filename"
                )
                or metadata.get(
                    "file_name"
                )
            )

            if not filename:
                continue

            page = (
                metadata.get(
                    "page_number"
                )
            )

            sheet = (
                metadata.get(
                    "sheet_name"
                )
            )

            if page is not None:
                citation = (
                    f"[Source: "
                    f"{filename}, "
                    f"Page {page}]"
                )

            elif sheet:
                citation = (
                    f"[Source: "
                    f"{filename}, "
                    f"Sheet: {sheet}]"
                )

            else:
                citation = (
                    f"[Source: "
                    f"{filename}]"
                )

            if citation in seen:
                continue

            seen.add(
                citation
            )

            result.append(
                citation
            )

        return result

    # =========================================================
    # DOCUMENT COUNT
    # =========================================================

    @staticmethod
    def _document_count(
        evidence: Sequence[Dict[str, Any]],
    ) -> int:

        document_keys: set[str] = set()

        for item in evidence:

            metadata = dict(
                item.get(
                    "metadata"
                )
                or {}
            )

            value = (
                metadata.get(
                    "document_id"
                )
                or metadata.get(
                    "filename"
                )
            )

            if value:
                document_keys.add(
                    str(
                        value
                    )
                )

        return len(
            document_keys
        )