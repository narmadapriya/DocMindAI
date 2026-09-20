from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Sequence

from app.core.logging import get_logger, log_event

logger = get_logger(__name__)


class VerificationAgent:
    """
    Phase 10 Verification Agent.

    Verifies:

        - evidence grounding
        - numeric claims
        - evidence IDs
        - conflict acknowledgement

    Verification supports:

        - normal textual answers
        - numeric-only answers
        - currency values
        - comma-formatted numbers
        - decimal values
        - percentages
        - mixed text + numeric answers

    Examples:

        "$32,500"
        "4,820"
        "11.5%"
        "The starting price is $32,500."
    """

    # =========================================================
    # VERIFY
    # =========================================================

    def verify(
        self,
        query: str,
        reasoning: Dict[str, Any],
        evidence: Sequence[
            Dict[str, Any]
        ],
        *,
        validation: Dict[str, Any]
        | None = None,
    ) -> Dict[str, Any]:

        # =====================================================
        # ANSWER
        # =====================================================

        answer = str(
            reasoning.get(
                "answer",
                "",
            )
        ).strip()

        if not answer:

            return self._fail(
                "empty_answer",
                query,
            )

        # =====================================================
        # EVIDENCE LOOKUP
        # =====================================================

        evidence_by_id = {
            str(
                item.get(
                    "chunk_id"
                )
            ): item

            for item in evidence

            if item.get(
                "chunk_id"
            )
        }

        # =====================================================
        # USED EVIDENCE IDS
        # =====================================================

        used_ids = [
            str(value)

            for value in (
                reasoning.get(
                    "used_evidence_ids",
                    [],
                )
                or []
            )
        ]

        # Remove duplicate IDs while preserving order.
        used_ids = list(
            dict.fromkeys(
                used_ids
            )
        )

        used_evidence = [
            evidence_by_id[
                evidence_id
            ]

            for evidence_id
            in used_ids

            if evidence_id
            in evidence_by_id
        ]

        source_kinds = self._source_kinds(used_evidence)
        all_source_kinds = self._source_kinds(evidence)
        structured_source = bool(
            all_source_kinds & {"csv", "xlsx"}
        )

        # -----------------------------------------------------
        # A reasoning answer must identify at least one real
        # retrieved evidence chunk.
        # -----------------------------------------------------

        if not used_evidence:

            return self._fail(
                "no_cited_evidence_ids",
                query,
            )

        # =====================================================
        # SOURCE TEXT
        # =====================================================

        source_text = " ".join(
            str(
                item.get(
                    "content",
                    "",
                )
            )

            for item
            in used_evidence
        ).strip()

        source_text_lower = (
            source_text.lower()
        )

        # CSV/XLSX analytical verification needs the complete retrieved table,
        # not only the one or two evidence IDs the small local model chose to
        # cite.  Ordinary grounding/citation validation still uses
        # ``used_evidence`` above.  DOCX/PDF remain unchanged.
        structured_evidence = [
            item
            for item in evidence
            if self._source_kind(item) in {"csv", "xlsx"}
        ]
        structured_source_text = " ".join(
            str(item.get("content", ""))
            for item in structured_evidence
        ).strip()

        # TXT-only verification may need facts from more than one retrieved
        # section. It still verifies only already-retrieved evidence and does
        # not alter retrieval/citation behavior. DOCX/CSV/XLSX/PDF text paths
        # remain unchanged.
        txt_source = "txt" in all_source_kinds
        txt_source_text = "\n".join(
            str(item.get("content", "") or "")
            for item in evidence
            if self._source_kind(item) == "txt"
        ).strip()

        # =====================================================
        # QUERY/EVIDENCE MODE
        # =====================================================

        visual_query = (
            self._is_visual_query(
                query
            )
        )

        visual_grounding_ok = (
            visual_query
            and any(
                self._visual_evidence_available(
                    item
                )
                for item in used_evidence
            )
        )

        calculation_query = (
            self._is_calculation_query(
                query
            )
        )

        # Derived calculations naturally introduce words such as
        # "increased", "approximately", and "percentage" that may not
        # occur verbatim in the source table. Keep the ordinary strict
        # threshold unchanged; use a slightly lower threshold only for
        # explicit arithmetic questions after numeric grounding succeeds.
        required_text_overlap = (
            0.10
            if calculation_query
            else 0.20
        )

        # =====================================================
        # TEXTUAL GROUNDING
        # =====================================================

        answer_tokens = (
            self._tokens(
                answer
            )
        )

        evidence_tokens = (
            self._tokens(
                source_text
            )
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Numeric-only answers such as:
        #
        #     $32,500
        #
        # have zero alphabetic tokens.
        #
        # Therefore textual overlap must not automatically
        # become zero and fail a correctly grounded numeric
        # response.
        # -----------------------------------------------------

        if answer_tokens:

            overlap = (
                len(
                    answer_tokens
                    & evidence_tokens
                )
                / max(
                    1,
                    len(
                        answer_tokens
                    ),
                )
            )

        else:

            # No meaningful text tokens exist.
            # Numeric grounding is evaluated independently
            # below.
            overlap = 1.0

        # =====================================================
        # NUMERIC GROUNDING
        # =====================================================

        answer_numbers = (
            self._extract_numbers(
                answer
            )
        )

        numeric_source_text = (
            structured_source_text
            if structured_source and structured_source_text
            else source_text
        )

        # TXT arithmetic questions may explicitly supply the operands in the
        # user's question while semantic top-k returns only one of the distant
        # supporting TXT sections. Treat those user-supplied operands as part of
        # the arithmetic verification context only for TXT calculation queries.
        # This does not relax grounding for ordinary factual answers or any
        # DOCX/CSV/XLSX/PDF request.
        if txt_source and calculation_query:
            numeric_source_text = (
                f"{numeric_source_text}\n{query}"
            ).strip()

        evidence_number_list = (
            self._extract_numbers(
                numeric_source_text
            )
        )

        evidence_numbers = set(
            evidence_number_list
        )

        if visual_grounding_ok:
            # The cited image/chart itself is the source of numeric truth.
            # A chart label or value may not exist in OCR/description text,
            # so text-only numeric matching must not reject a correctly
            # cited multimodal answer. This branch is allowed only when the
            # query is explicitly visual and the cited evidence points to a
            # real local image file.
            unsupported_numbers = []
        else:
            unsupported_numbers = [
                number

                for number
                in answer_numbers

                if (
                    number
                    not in evidence_numbers
                    and not self._derived_number_supported(
                        number=number,
                        evidence_numbers=evidence_number_list,
                        query=query,
                    )
                )
            ]

        numeric_grounding_ok = (
            not unsupported_numbers
        )

        # -----------------------------------------------------
        # If the answer contains only numeric content,
        # at least one supported number must exist.
        # -----------------------------------------------------

        numeric_only_answer = (
            not answer_tokens
            and bool(
                answer_numbers
            )
        )

        if numeric_only_answer:

            grounding_ok = (
                numeric_grounding_ok
                and bool(
                    answer_numbers
                )
            )

        else:

            grounding_ok = (
                visual_grounding_ok
                or (
                    overlap >= required_text_overlap
                    and numeric_grounding_ok
                )
            )

        # =====================================================
        # CLAIM-LEVEL SUPPORT
        # =====================================================

        claims = [
            str(claim).strip()

            for claim in (
                reasoning.get(
                    "claims",
                    []
                )
                or []
            )

            if str(
                claim
            ).strip()
        ]

        unsupported_claims: list[
            str
        ] = []

        # -----------------------------------------------------
        # Claims provide a stronger grounding signal than a
        # short answer such as "$32,500".
        #
        # We only reject a claim when:
        #
        #   - it contains unsupported numeric values, OR
        #   - it has meaningful text but essentially no
        #     overlap with cited evidence.
        # -----------------------------------------------------

        for claim in claims:

            claim_tokens = (
                self._tokens(
                    claim
                )
            )

            claim_numbers = (
                self._extract_numbers(
                    claim
                )
            )

            if visual_grounding_ok:
                unsupported_claim_numbers = []
            else:
                unsupported_claim_numbers = [
                    number

                    for number
                    in claim_numbers

                    if (
                        number
                        not in evidence_numbers
                        and not self._derived_number_supported(
                            number=number,
                            evidence_numbers=evidence_number_list,
                            query=query,
                        )
                    )
                ]

            if unsupported_claim_numbers:

                unsupported_claims.append(
                    claim
                )

                continue

            if claim_tokens and not visual_grounding_ok:

                claim_overlap = (
                    len(
                        claim_tokens
                        & evidence_tokens
                    )
                    / max(
                        1,
                        len(
                            claim_tokens
                        ),
                    )
                )

                if claim_overlap < required_text_overlap:

                    unsupported_claims.append(
                        claim
                    )

        claims_grounded = (
            not unsupported_claims
        )

        # =====================================================
        # DIRECT ANSWER SUPPORT
        # =====================================================

        direct_support = (
            self._direct_support(
                answer=answer,
                source_text=source_text,
            )
        )

        # -----------------------------------------------------
        # A directly quoted/contained answer such as:
        #
        #     $32,500
        #
        # should count as strongly grounded even if it has no
        # alphabetic tokens.
        # -----------------------------------------------------

        if (
            direct_support
            and numeric_grounding_ok
        ):

            grounding_ok = True

        # =====================================================
        # CONFLICT VALIDATION
        # =====================================================

        conflicts = (
            validation
            or {}
        ).get(
            "conflicts",
            [],
        )

        conflict_acknowledged = (
            not conflicts

            or any(
                word
                in answer.lower()

                for word
                in (
                    "conflict",
                    "different",
                    "disagree",
                    "respectively",
                    "varies",
                    "vary",
                )
            )
        )

        # =====================================================
        # QUERY COMPLETENESS
        # =====================================================

        query_complete = (
            self._query_complete(
                query=query,
                answer=answer,
            )
        )

        # =====================================================
        # ORDERED / PREVIOUS-ROW RELATIONSHIP
        # =====================================================

        sequence_relation_ok = (
            self._sequence_relation_supported(
                query=query,
                answer=answer,
                source_text=(
                    structured_source_text
                    if structured_source and structured_source_text
                    else source_text
                ),
            )
        )

        # =====================================================
        # CSV/XLSX STRUCTURED RELATION / AGGREGATION CHECKS
        # =====================================================

        structured_logic_ok = True

        if structured_source:
            structured_logic_ok = (
                self._structured_query_supported(
                    query=query,
                    answer=answer,
                    source_text=(structured_source_text or source_text),
                )
            )

        # =====================================================
        # TXT-ONLY ARITHMETIC / COMPLETENESS CHECK
        # =====================================================

        txt_logic_ok = True
        if txt_source:
            txt_logic_ok = self._txt_query_supported(
                query=query,
                answer=answer,
                source_text=(txt_source_text or source_text),
            )

        # =====================================================
        # PDF-VISUAL-ONLY RELATION / COMPLETENESS CHECK
        # =====================================================

        pdf_visual_logic_ok = True
        if visual_query and "pdf" in all_source_kinds:
            pdf_visual_logic_ok = self._pdf_visual_query_supported(
                query=query,
                answer=answer,
            )

        # =====================================================
        # PDF-TEXT-ONLY EXTREMUM / EXPLICIT-LIST CHECKS
        #
        # Isolated to non-visual PDF questions so the already-working
        # DOCX/CSV/TXT/XLSX and PDF visual paths are unchanged.
        # =====================================================

        pdf_text_logic_ok = True
        if (
            "pdf" in all_source_kinds
            and not visual_query
        ):
            pdf_text_logic_ok = self._pdf_text_query_supported(
                query=query,
                answer=answer,
                evidence=evidence,
            )

        # =====================================================
        # FINAL PASS
        # =====================================================

        passed = (
            grounding_ok
            and claims_grounded
            and conflict_acknowledged
            and query_complete
            and sequence_relation_ok
            and structured_logic_ok
            and txt_logic_ok
            and pdf_visual_logic_ok
            and pdf_text_logic_ok
        )

        # =====================================================
        # FAILURE REASON
        # =====================================================

        if passed:

            reason = (
                "grounded"
            )

        elif unsupported_numbers:

            reason = (
                "unsupported_numeric_claim"
            )

        elif unsupported_claims:

            reason = (
                "unsupported_claim"
            )

        elif not conflict_acknowledged:

            reason = (
                "unacknowledged_conflict"
            )

        elif not sequence_relation_ok:

            reason = (
                "unsupported_ordered_sequence_claim"
            )

        elif not structured_logic_ok:

            reason = (
                "unsupported_structured_relation"
            )

        elif not txt_logic_ok:

            reason = (
                "unsupported_txt_arithmetic_or_completeness"
            )

        elif not pdf_visual_logic_ok:

            reason = (
                "unsupported_pdf_visual_relation"
            )

        elif not pdf_text_logic_ok:

            reason = (
                "unsupported_pdf_text_extremum_or_list"
            )

        elif not query_complete:

            reason = (
                "incomplete_answer"
            )

        else:

            reason = (
                "insufficient_grounding"
            )

        # =====================================================
        # RESPONSE
        # =====================================================

        log_event(
            logger,
            "verification",
            status=("pass" if passed else "fail"),
            reason=reason,
            evidence_count=len(evidence),
            used_evidence_count=len(used_ids),
        )

        return {
            "status": (
                "pass"
                if passed
                else "fail"
            ),

            "passed":
                passed,

            "reason":
                reason,

            "grounding_overlap":
                round(
                    overlap,
                    4,
                ),

            "direct_support":
                direct_support,

            "numeric_only_answer":
                numeric_only_answer,

            "numeric_grounding_ok":
                numeric_grounding_ok,

            "answer_numbers":
                answer_numbers,

            "unsupported_numbers":
                unsupported_numbers,

            "claims_grounded":
                claims_grounded,

            "unsupported_claims":
                unsupported_claims,

            "used_evidence_ids":
                used_ids,

            "conflict_acknowledged":
                conflict_acknowledged,

            "visual_grounding_ok":
                visual_grounding_ok,

            "sequence_relation_ok":
                sequence_relation_ok,

            "structured_logic_ok":
                structured_logic_ok,

            "txt_logic_ok":
                txt_logic_ok,

            "pdf_visual_logic_ok":
                pdf_visual_logic_ok,

            "pdf_text_logic_ok":
                pdf_text_logic_ok,

            "retry_required":
                not passed,

            "rewritten_query": (
                self.rewrite_query(
                    query,
                    {},
                )
                if not passed
                else ""
            ),
        }

    # =========================================================
    # QUERY REWRITE
    # =========================================================

    def rewrite_query(
        self,
        query: str,
        verification: Dict[
            str,
            Any,
        ],
    ) -> str:
        """
        Rewrite a failed query for one retrieval retry.

        Avoid repeatedly appending the same retry instruction.
        """

        base_query = str(
            query
            or ""
        ).strip()

        instruction = (
            "Use only directly supported facts "
            "from the retrieved evidence; include "
            "source-specific values and resolve or "
            "explicitly report conflicts."
        )

        # -----------------------------------------------------
        # Prevent:
        #
        # question + instruction + instruction + instruction
        #
        # across verification retries.
        # -----------------------------------------------------

        if (
            instruction.lower()
            in base_query.lower()
        ):

            return base_query

        return (
            f"{base_query} "
            f"{instruction}"
        ).strip()

    # =========================================================
    # QUERY MODE HELPERS
    # =========================================================

    @staticmethod
    def _is_visual_query(
        query: str,
    ) -> bool:
        q = str(
            query
            or ""
        ).lower()

        return any(
            term in q
            for term in (
                "image",
                "photo",
                "picture",
                "visual",
                "chart",
                "graph",
                "plot",
                "diagram",
                "axis",
            )
        )

    @staticmethod
    def _is_calculation_query(
        query: str,
    ) -> bool:
        q = str(
            query
            or ""
        ).lower()

        return any(
            term in q
            for term in (
                "increase",
                "increased",
                "decrease",
                "decreased",
                "difference",
                "change",
                "changed",
                "percentage",
                "percent",
                "%",
                "sum",
                "combined",
                "together",
                "average",
                "mean",
                "remaining",
                "remainder",
            )
        )

    @classmethod
    def _visual_evidence_available(
        cls,
        evidence: Dict[str, Any],
    ) -> bool:
        """
        Confirm that cited visual evidence resolves to a genuine local
        image asset. This is intentionally stricter than checking only
        chunk_type so a normal text/table chunk cannot bypass numeric
        verification merely because the user mentioned a chart.
        """

        metadata = (
            evidence.get("metadata")
            or {}
        )

        chunk_type = str(
            metadata.get(
                "chunk_type",
                "",
            )
        ).lower()

        if chunk_type not in {
            "image",
            "chart",
        }:
            return False

        for key in (
            "image_path",
            "source_path",
            "path",
            "source_location",
        ):
            value = metadata.get(key)

            if not value:
                continue

            path = Path(
                str(value)
            )

            if (
                path.exists()
                and path.is_file()
                and path.suffix.lower()
                in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                }
            ):
                return True

        return False

    # =========================================================
    # ORDERED SEQUENCE SUPPORT
    # =========================================================

    @classmethod
    def _sequence_relation_supported(
        cls,
        *,
        query: str,
        answer: str,
        source_text: str,
    ) -> bool:
        """
        Protect ordered table questions such as:

            "Which month decreased compared with the previous month?"

        The previous verifier accepted any pairwise arithmetic difference
        from the table. That could incorrectly validate an answer like
        "June, by 100" because 100 happened to be the difference between
        two unrelated rows elsewhere in the evidence.

        For explicit previous-month/previous-row questions, require the
        named row to be compared with its immediately preceding row.
        Other queries are unchanged.
        """

        q = re.sub(
            r"\s+",
            " ",
            str(query or "").lower(),
        ).strip()

        if not any(
            phrase in q
            for phrase in (
                "previous month",
                "previous row",
                "preceding month",
                "prior month",
            )
        ):
            return True

        rows = cls._extract_month_unit_rows(
            source_text
        )

        if len(rows) < 2:
            return False

        answer_month = cls._month_from_text(
            answer
        )

        if answer_month is None:
            return False

        current_index = None

        for index, row in enumerate(rows):
            if row[0] == answer_month:
                current_index = index
                break

        if (
            current_index is None
            or current_index <= 0
        ):
            return False

        previous_value = rows[
            current_index - 1
        ][1]
        current_value = rows[
            current_index
        ][1]

        delta = (
            current_value
            - previous_value
        )

        asks_decrease = any(
            term in q
            for term in (
                "decrease",
                "decreased",
                "decline",
                "declined",
                "lower",
                "fell",
                "drop",
                "dropped",
            )
        )

        asks_increase = any(
            term in q
            for term in (
                "increase",
                "increased",
                "higher",
                "rose",
                "growth",
            )
        )

        if asks_decrease and delta >= 0:
            return False

        if asks_increase and delta <= 0:
            return False

        expected = abs(delta)

        answer_numbers = []

        for raw in cls._extract_numbers(
            answer
        ):
            try:
                value = Decimal(
                    str(raw)
                )
            except InvalidOperation:
                continue

            if cls._is_year_value(value):
                continue

            answer_numbers.append(value)

        return any(
            cls._decimal_close(
                value,
                expected,
            )
            for value in answer_numbers
        )

    @classmethod
    def _extract_month_unit_rows(
        cls,
        source_text: str,
    ) -> list[tuple[int, Decimal]]:
        """
        Extract ordered Month -> Units Sold rows from the structured
        formats already produced by DocMindAI's CSV/table parser.

        Supported representations include:
            Jan | 2025 | 1050 | ...
            Jan,2025,1050,...
            Row 2: Month: Jan (January); ...; Units Sold: 1050; ...
        """

        rows: list[
            tuple[int, Decimal]
        ] = []
        seen: set[tuple[int, str]] = set()

        text = str(
            source_text
            or ""
        )

        labelled = re.compile(
            r"Month\s*:\s*"
            r"(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"(?:\s*\([A-Za-z]+\))?"
            r".*?"
            r"Units\s+Sold\s*:\s*"
            r"(?P<units>[\d,]+(?:\.\d+)?)",
            re.IGNORECASE,
        )

        for match in labelled.finditer(text):
            month = cls._month_number(
                match.group("month")
            )

            if month is None:
                continue

            try:
                units = Decimal(
                    match.group("units").replace(",", "")
                )
            except InvalidOperation:
                continue

            marker = (
                month,
                cls._normalize_decimal(units),
            )

            if marker not in seen:
                seen.add(marker)
                rows.append((month, units))

        # Table/CSV rows.
        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            delimiter = (
                "|"
                if "|" in stripped
                else ","
                if "," in stripped
                else None
            )

            if delimiter is None:
                continue

            parts = [
                part.strip()
                for part in stripped.split(delimiter)
            ]

            if len(parts) < 3:
                continue

            month = cls._month_number(
                parts[0]
            )

            if month is None:
                continue

            # DocMindAI's CSV/XLSX sales representation places Year in
            # column 2 and Units Sold in column 3.
            try:
                units = Decimal(
                    re.sub(
                        r"[^0-9.+-]",
                        "",
                        parts[2],
                    )
                )
            except InvalidOperation:
                continue

            marker = (
                month,
                cls._normalize_decimal(units),
            )

            if marker not in seen:
                seen.add(marker)
                rows.append((month, units))

        # Preserve document order where possible. If evidence chunks were
        # concatenated in a different retrieval order, normalize monthly
        # records to calendar order so "previous month" remains deterministic.
        rows.sort(
            key=lambda item: item[0]
        )

        return rows

    @classmethod
    def _month_from_text(
        cls,
        text: str,
    ) -> int | None:
        match = re.search(
            r"\b("
            r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
            r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
            r")\b",
            str(text or ""),
            re.IGNORECASE,
        )

        if not match:
            return None

        return cls._month_number(
            match.group(1)
        )

    @staticmethod
    def _month_number(
        value: str,
    ) -> int | None:
        key = str(
            value
            or ""
        ).strip().lower()[:3]

        return {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }.get(key)

    # =========================================================
    # DIRECT SUPPORT
    # =========================================================

    @classmethod
    def _direct_support(
        cls,
        *,
        answer: str,
        source_text: str,
    ) -> bool:
        """
        Check whether the answer itself appears directly in
        cited evidence after harmless formatting normalization.

        Handles:

            "$32,500"
                vs
            "Starting at $32,500"

        and:

            "4,820"
                vs
            "FY2025 units sold: 4,820"
        """

        normalized_answer = (
            cls._normalize_text(
                answer
            )
        )

        normalized_source = (
            cls._normalize_text(
                source_text
            )
        )

        if not normalized_answer:

            return False

        if (
            normalized_answer
            in normalized_source
        ):

            return True

        # -----------------------------------------------------
        # Numeric-only answers may differ only by commas or
        # currency formatting.
        # -----------------------------------------------------

        answer_numbers = (
            cls._extract_numbers(
                answer
            )
        )

        if (
            answer_numbers
            and not cls._tokens(
                answer
            )
        ):

            source_numbers = set(
                cls._extract_numbers(
                    source_text
                )
            )

            return all(
                number
                in source_numbers

                for number
                in answer_numbers
            )

        return False

    # =========================================================
    # NUMBER EXTRACTION
    # =========================================================

    @classmethod
    def _extract_numbers(
        cls,
        text: str,
    ) -> list[str]:
        """
        Extract and normalize numbers.

        Examples:

            "$32,500"
                -> "32500"

            "4,820"
                -> "4820"

            "11.5%"
                -> "11.5"

            "$181.1 million"
                -> "181100000"

            "2025"
                -> "2025"

        Scale words are normalized so an answer such as
        ``$181.1 million`` can be verified against evidence containing
        ``$181,100,000`` without weakening numeric grounding.
        """

        pattern = re.compile(
            r"(?<!\w)"
            r"(?P<number>"
            r"[-+]?"
            r"(?:"
            r"\d{1,3}(?:,\d{3})+"
            r"|"
            r"\d+"
            r")"
            r"(?:\.\d+)?"
            r")"
            r"(?P<percent>%?)"
            r"(?:\s*(?P<scale>thousand|million|billion))?",
            re.IGNORECASE,
        )

        multipliers = {
            "thousand": Decimal("1000"),
            "million": Decimal("1000000"),
            "billion": Decimal("1000000000"),
        }

        normalized: list[str] = []

        for match in pattern.finditer(
            str(text or "")
        ):
            cleaned = (
                match.group("number")
                .replace(",", "")
                .strip()
            )

            if not cleaned:
                continue

            try:
                decimal_value = Decimal(
                    cleaned
                )
            except InvalidOperation:
                continue

            scale = (
                match.group("scale")
                or ""
            ).lower()

            if scale:
                decimal_value *= multipliers[
                    scale
                ]

            normalized.append(
                cls._normalize_decimal(
                    decimal_value
                )
            )

        return normalized

    @staticmethod
    def _normalize_decimal(
        value: Decimal,
    ) -> str:
        normalized_value = format(
            value.normalize(),
            "f",
        )

        if "." in normalized_value:
            normalized_value = (
                normalized_value
                .rstrip("0")
                .rstrip(".")
            )

        if normalized_value in {
            "-0",
            "+0",
        }:
            return "0"

        return normalized_value

    # =========================================================
    # DERIVED NUMERIC SUPPORT
    # =========================================================

    @classmethod
    def _derived_number_supported(
        cls,
        *,
        number: str,
        evidence_numbers: Sequence[str],
        query: str,
    ) -> bool:
        """
        Permit only simple deterministic arithmetic when the query
        explicitly asks for a calculation.

        This preserves strict direct numeric grounding for ordinary
        factual questions while allowing answers such as:

            1,480 - 1,050 = 430
            430 / 1,050 * 100 ~= 41.0%

        No model-generated value is accepted merely because it is
        plausible; it must match a supported arithmetic result from
        cited evidence.
        """

        q = str(query or "").lower()

        asks_difference = any(
            term in q
            for term in (
                "increase",
                "increased",
                "decrease",
                "decreased",
                "difference",
                "change",
                "changed",
                "more than",
                "less than",
                "fewer than",
                "remaining",
                "remainder",
            )
        )

        asks_percentage = any(
            term in q
            for term in (
                "percentage",
                "percent",
                "%",
            )
        )

        asks_sum = any(
            term in q
            for term in (
                "sum",
                "combined",
                "together",
            )
        )

        asks_average = any(
            term in q
            for term in (
                "average",
                "mean",
            )
        )

        if not (
            asks_difference
            or asks_percentage
            or asks_sum
            or asks_average
        ):
            return False

        try:
            target = Decimal(
                str(number)
            )
        except InvalidOperation:
            return False

        values: list[Decimal] = []
        seen: set[str] = set()

        for raw in evidence_numbers:
            try:
                value = Decimal(
                    str(raw)
                )
            except InvalidOperation:
                continue

            if cls._is_year_value(value):
                continue

            marker = cls._normalize_decimal(
                value
            )

            if marker in seen:
                continue

            seen.add(marker)
            values.append(value)

        # Keep pairwise validation bounded even for a large retrieved
        # table. Normal RAG evidence is far below this limit.
        values = values[:80]

        if len(values) < 2:
            return False

        for index, first in enumerate(values):
            for second in values[index + 1 :]:
                candidates: list[Decimal] = []

                if asks_difference:
                    candidates.extend(
                        (
                            first - second,
                            second - first,
                            abs(first - second),
                        )
                    )

                if asks_sum:
                    candidates.append(
                        first + second
                    )

                if asks_average:
                    candidates.append(
                        (first + second)
                        / Decimal("2")
                    )

                if asks_percentage:
                    delta = abs(
                        first - second
                    )

                    if first != 0:
                        candidates.append(
                            delta
                            / abs(first)
                            * Decimal("100")
                        )

                    if second != 0:
                        candidates.append(
                            delta
                            / abs(second)
                            * Decimal("100")
                        )

                if any(
                    cls._decimal_close(
                        target,
                        candidate,
                    )
                    for candidate in candidates
                ):
                    return True

        return False

    @staticmethod
    def _is_year_value(
        value: Decimal,
    ) -> bool:
        if value != value.to_integral_value():
            return False

        return (
            Decimal("1900")
            <= abs(value)
            <= Decimal("2100")
        )

    @staticmethod
    def _decimal_close(
        left: Decimal,
        right: Decimal,
    ) -> bool:
        difference = abs(
            left - right
        )

        tolerance = max(
            Decimal("0.05"),
            abs(right)
            * Decimal("0.0025"),
        )

        return difference <= tolerance

    # =========================================================
    # CSV/XLSX STRUCTURED LOGIC
    # =========================================================

    @classmethod
    def _source_kind(cls, item: Dict[str, Any]) -> str:
        metadata = item.get("metadata") or {}
        filename = str(
            metadata.get("filename")
            or metadata.get("file_name")
            or metadata.get("original_filename")
            or ""
        ).lower()

        if filename.endswith(".csv"):
            return "csv"
        if filename.endswith((".xlsx", ".xls")):
            return "xlsx"
        if filename.endswith(".pdf"):
            return "pdf"
        if filename.endswith(".docx"):
            return "docx"
        if filename.endswith(".txt"):
            return "txt"

        file_type = str(metadata.get("file_type", "")).lower()
        if "csv" in file_type:
            return "csv"
        if "xlsx" in file_type or "excel" in file_type or "spreadsheet" in file_type:
            return "xlsx"
        if "pdf" in file_type:
            return "pdf"
        if "docx" in file_type or "word" in file_type:
            return "docx"
        if "txt" in file_type or "text/plain" in file_type:
            return "txt"
        return ""

    @classmethod
    def _source_kinds(
        cls,
        evidence: Sequence[Dict[str, Any]],
    ) -> set[str]:
        return {
            kind
            for kind in (cls._source_kind(item) for item in evidence)
            if kind
        }

    @classmethod
    def _structured_query_supported(
        cls,
        *,
        query: str,
        answer: str,
        source_text: str,
    ) -> bool:
        """Validate only CSV/XLSX relationships that a 3B model often misreads."""

        q = re.sub(r"\s+", " ", str(query or "").lower()).strip()

        if any(
            phrase in q
            for phrase in (
                "previous month",
                "previous row",
                "preceding month",
                "prior month",
            )
        ):
            # The existing ordered-sequence verifier is authoritative.
            return cls._sequence_relation_supported(
                query=query,
                answer=answer,
                source_text=source_text,
            )

        if (
            "from" in q
            and "to" in q
            and any(term in q for term in ("increase", "decrease", "change"))
        ):
            return cls._endpoint_change_supported(
                query=query,
                answer=answer,
                source_text=source_text,
            )

        if any(term in q for term in ("highest", "maximum", "largest")):
            if "revenue" in q:
                return cls._revenue_extremum_supported(
                    query=query,
                    answer=answer,
                    source_text=source_text,
                    highest=True,
                )

        if any(term in q for term in ("lowest", "minimum", "smallest")):
            if "revenue" in q:
                return cls._revenue_extremum_supported(
                    query=query,
                    answer=answer,
                    source_text=source_text,
                    highest=False,
                )

        if (
            "in production" in q
            and "current stock" in q
            and any(term in q for term in ("more", "greater", "higher"))
        ):
            return cls._inventory_inequality_supported(
                query=query,
                answer=answer,
                source_text=source_text,
            )

        return True

    @classmethod
    def _extract_sales_rows(
        cls,
        source_text: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, str, str]] = set()

        for line in str(source_text or "").splitlines():
            if "|" not in line:
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 5:
                continue

            month = cls._month_number(parts[0])
            if month is None:
                continue

            try:
                units = Decimal(re.sub(r"[^0-9.+-]", "", parts[2]))
                revenue = Decimal(re.sub(r"[^0-9.+-]", "", parts[4]))
            except InvalidOperation:
                continue

            marker = (
                month,
                cls._normalize_decimal(units),
                cls._normalize_decimal(revenue),
            )
            if marker in seen:
                continue
            seen.add(marker)
            rows.append(
                {
                    "month": month,
                    "month_text": parts[0],
                    "units": units,
                    "revenue": revenue,
                }
            )

        # Record-text fallback.
        labelled = re.compile(
            r"Month\s*:\s*(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r".*?Units\s+Sold\s*:\s*(?P<units>[\d,]+(?:\.\d+)?)"
            r".*?Revenue(?:\s*\(USD\))?\s*:\s*\$?(?P<revenue>[\d,]+(?:\.\d+)?)",
            re.IGNORECASE | re.DOTALL,
        )

        for match in labelled.finditer(str(source_text or "")):
            month = cls._month_number(match.group("month"))
            if month is None:
                continue
            try:
                units = Decimal(match.group("units").replace(",", ""))
                revenue = Decimal(match.group("revenue").replace(",", ""))
            except InvalidOperation:
                continue
            marker = (
                month,
                cls._normalize_decimal(units),
                cls._normalize_decimal(revenue),
            )
            if marker in seen:
                continue
            seen.add(marker)
            rows.append(
                {
                    "month": month,
                    "month_text": match.group("month"),
                    "units": units,
                    "revenue": revenue,
                }
            )

        rows.sort(key=lambda row: row["month"])
        return rows

    @classmethod
    def _query_month_range(cls, query: str) -> tuple[int | None, int | None]:
        months = [
            cls._month_number(value)
            for value in re.findall(
                r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
                str(query or ""),
                re.IGNORECASE,
            )
        ]
        months = [value for value in months if value is not None]
        if not months:
            return None, None
        if len(months) == 1:
            return months[0], months[0]
        return months[0], months[1]

    @classmethod
    def _revenue_extremum_supported(
        cls,
        *,
        query: str,
        answer: str,
        source_text: str,
        highest: bool,
    ) -> bool:
        rows = cls._extract_sales_rows(source_text)
        if not rows:
            return False

        start, end = cls._query_month_range(query)
        if start is not None and end is not None:
            low, high = sorted((start, end))
            scoped = [row for row in rows if low <= row["month"] <= high]
            if scoped:
                rows = scoped

        target = (
            max(rows, key=lambda row: row["revenue"])
            if highest
            else min(rows, key=lambda row: row["revenue"])
        )

        answer_month = cls._month_from_text(answer)
        if answer_month != target["month"]:
            return False

        answer_numbers = cls._extract_numbers(answer)
        normalized = {
            cls._normalize_decimal(Decimal(value))
            for value in answer_numbers
            if not cls._is_year_number_string(value)
        }

        return (
            cls._normalize_decimal(target["revenue"]) in normalized
            or cls._normalize_decimal(target["units"]) in normalized
        )

    @classmethod
    def _endpoint_change_supported(
        cls,
        *,
        query: str,
        answer: str,
        source_text: str,
    ) -> bool:
        rows = cls._extract_sales_rows(source_text)
        if len(rows) < 2:
            return False

        start_month, end_month = cls._query_month_range(query)
        if start_month is None or end_month is None:
            return True

        by_month = {row["month"]: row for row in rows}
        if start_month not in by_month or end_month not in by_month:
            return False

        start_units = by_month[start_month]["units"]
        end_units = by_month[end_month]["units"]
        delta = end_units - start_units

        if start_units == 0:
            return False

        percent = (delta / start_units) * Decimal("100")

        values: list[Decimal] = []
        for raw in cls._extract_numbers(answer):
            try:
                value = Decimal(str(raw))
            except InvalidOperation:
                continue
            if cls._is_year_value(value):
                continue
            values.append(value)

        has_delta = any(cls._decimal_close(value, abs(delta)) for value in values)
        has_percent = any(
            abs(value - abs(percent)) <= Decimal("0.2")
            for value in values
        )

        q = str(query or "").lower()
        if "percentage" in q or "percent" in q or "%" in q:
            return has_delta and has_percent
        return has_delta

    @classmethod
    def _extract_inventory_rows(
        cls,
        source_text: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        for line in str(source_text or "").splitlines():
            if "|" not in line:
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 4:
                continue
            model = parts[0]
            if not model.lower().startswith("apex "):
                continue
            try:
                current = Decimal(re.sub(r"[^0-9.+-]", "", parts[2]))
                production = Decimal(re.sub(r"[^0-9.+-]", "", parts[3]))
            except InvalidOperation:
                continue
            key = model.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "model": model,
                    "current": current,
                    "production": production,
                }
            )

        labelled = re.compile(
            r"Model\s*:\s*(?P<model>Apex[^;\n]+).*?Current\s+Stock\s*:\s*(?P<current>[\d,]+).*?In\s+Production\s*:\s*(?P<production>[\d,]+)",
            re.IGNORECASE | re.DOTALL,
        )
        for match in labelled.finditer(str(source_text or "")):
            model = match.group("model").strip()
            key = model.lower()
            if key in seen:
                continue
            try:
                current = Decimal(match.group("current").replace(",", ""))
                production = Decimal(match.group("production").replace(",", ""))
            except InvalidOperation:
                continue
            seen.add(key)
            rows.append(
                {
                    "model": model,
                    "current": current,
                    "production": production,
                }
            )

        return rows

    @classmethod
    def _inventory_inequality_supported(
        cls,
        *,
        query: str,
        answer: str,
        source_text: str,
    ) -> bool:
        rows = cls._extract_inventory_rows(source_text)
        if not rows:
            return False

        qualifying = [
            row for row in rows
            if row["production"] > row["current"]
        ]
        if not qualifying:
            return False

        answer_lower = str(answer or "").lower()
        named_rows = [
            row for row in rows
            if row["model"].lower() in answer_lower
        ]

        if not named_rows:
            return False

        qualifying_names = {row["model"].lower() for row in qualifying}
        if any(row["model"].lower() not in qualifying_names for row in named_rows):
            return False

        # If exactly one row satisfies the relationship, the answer must name it.
        if len(qualifying) == 1 and qualifying[0]["model"].lower() not in answer_lower:
            return False

        q = str(query or "").lower()
        if "both values" in q or "two values" in q:
            target = qualifying[0] if len(qualifying) == 1 else named_rows[0]
            answer_numbers = {
                cls._normalize_decimal(Decimal(value))
                for value in cls._extract_numbers(answer)
                if not cls._is_year_number_string(value)
            }
            if cls._normalize_decimal(target["current"]) not in answer_numbers:
                return False
            if cls._normalize_decimal(target["production"]) not in answer_numbers:
                return False

        return True

    # =========================================================
    # QUERY COMPLETENESS
    # =========================================================

    # =========================================================
    # TXT-ONLY ARITHMETIC / COMPLETENESS
    # =========================================================

    @classmethod
    def _txt_query_supported(
        cls,
        *,
        query: str,
        answer: str,
        source_text: str,
    ) -> bool:
        """
        Narrow TXT-only guard for the two failure modes observed in testing:
        derived remainder arithmetic and explicit multi-model field coverage.
        It does not run for DOCX/CSV/XLSX/PDF.
        """
        q = re.sub(r"\s+", " ", str(query or "")).strip()
        q_lower = q.lower()
        a = re.sub(r"\s+", " ", str(answer or "")).strip()
        a_lower = a.lower()

        # Reject Python/dict-style partial TXT answers for these quantitative
        # tasks; the user-facing answer must be complete natural language.
        if a.startswith("{") and a.endswith("}"):
            return False

        # Remaining-period arithmetic. Validate that the question operands are
        # present in the retrieved TXT evidence before checking the derived answer.
        if "remaining" in q_lower:
            unit_values = [
                int(value.replace(",", ""))
                for value in re.findall(r"([0-9][0-9,]*)\s+units?\b", q, re.IGNORECASE)
            ]
            money_values = [
                int(value.replace(",", ""))
                for value in re.findall(r"\$([0-9][0-9,]*)", q)
            ]

            if len(unit_values) >= 2 and len(money_values) >= 2:
                source_numbers = set(cls._extract_numbers(source_text))
                query_numbers = set(cls._extract_numbers(q))
                operands = {
                    str(unit_values[0]), str(unit_values[1]),
                    str(money_values[0]), str(money_values[1]),
                }

                # For this narrow TXT remainder shape, the user has explicitly
                # supplied the four operands in the question. They are therefore
                # valid arithmetic inputs even when semantic retrieval returns
                # only one of the source sections in top-k. At least one scoped
                # TXT evidence chunk is still required by the normal verifier.
                if not operands.issubset(source_numbers | query_numbers):
                    return False

                expected_units = unit_values[1] - unit_values[0]
                expected_revenue = money_values[1] - money_values[0]
                if expected_units < 0 or expected_revenue < 0:
                    return False

                answer_numbers = set(cls._extract_numbers(a))
                if str(expected_units) not in answer_numbers:
                    return False
                if str(expected_revenue) not in answer_numbers:
                    return False

                # Make sure both requested quantity types are actually stated.
                if "unit" not in a_lower:
                    return False
                if not ("$" in a or "revenue" in a_lower):
                    return False

        # Top-selling + fastest-growth multi-model completeness. Derive the two
        # model identities from narrative source text, then require units + price
        # for both when those table rows are present in retrieved evidence.
        if (
            "top-selling" in q_lower
            and "fastest" in q_lower
            and "starting price" in q_lower
            and "unit" in q_lower
        ):
            top_match = re.search(
                r"(Apex\s+.+?)\s+was\s+the\s+company'?s\s+top-selling\s+model",
                source_text,
                re.IGNORECASE,
            )
            fast_match = re.search(
                r"(Apex\s+.+?)\s+posted\s+the\s+fastest\s+year-over-year\s+growth",
                source_text,
                re.IGNORECASE,
            )

            if top_match and fast_match:
                top_name = re.sub(r"\s+", " ", top_match.group(1)).strip()
                fast_name = re.sub(r"\s+", " ", fast_match.group(1)).strip()
                top_row = cls._txt_model_metrics(top_name, source_text)
                fast_row = cls._txt_model_metrics(fast_name, source_text)

                if top_row and fast_row:
                    if top_name.lower() not in a_lower or fast_name.lower() not in a_lower:
                        return False

                    answer_numbers = set(cls._extract_numbers(a))
                    required = {
                        str(top_row["price"]), str(top_row["units"]),
                        str(fast_row["price"]), str(fast_row["units"]),
                    }
                    if not required.issubset(answer_numbers):
                        return False

        return True

    @staticmethod
    def _txt_model_metrics(
        model_name: str,
        text: str,
    ) -> Dict[str, int] | None:
        name_lower = str(model_name or "").lower()
        for raw_line in str(text or "").splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if name_lower not in line.lower() or "$" not in line:
                continue

            money = re.search(r"\$([0-9][0-9,]*)", line)
            if not money:
                continue

            tail = line[money.end():]
            numbers = re.findall(r"(?<![0-9])([0-9][0-9,]*)(?![0-9])", tail)
            if not numbers:
                continue

            try:
                return {
                    "price": int(money.group(1).replace(",", "")),
                    "units": int(numbers[-1].replace(",", "")),
                }
            except ValueError:
                continue

        return None

    # =========================================================
    # PDF-TEXT-ONLY EXTREMUM / EXPLICIT-LIST CHECKS
    # =========================================================

    @classmethod
    def _pdf_text_query_supported(
        cls,
        *,
        query: str,
        answer: str,
        evidence: Sequence[Dict[str, Any]],
    ) -> bool:
        q = re.sub(r"\s+", " ", str(query or "").lower()).strip()

        # PDF table extrema: verify the selected row and the requested metric
        # directly from the complete retrieved PDF table.
        if any(term in q for term in ("highest", "lowest", "maximum", "minimum", "largest", "smallest")):
            table_check = cls._pdf_table_extremum_supported(
                query=query,
                answer=answer,
                evidence=evidence,
            )
            if table_check is not None:
                return table_check

        # PDF explicit factor/driver/reason lists: when an explicit list is
        # present in the retrieved page evidence, require the answer to cover
        # every list member rather than accepting nearby unrelated facts.
        if any(term in q for term in ("factors", "drivers", "reasons", "causes", "priorities")):
            list_check = cls._pdf_explicit_list_supported(
                query=query,
                answer=answer,
                evidence=evidence,
            )
            if list_check is not None:
                return list_check

        return True

    @classmethod
    def _pdf_table_extremum_supported(
        cls,
        *,
        query: str,
        answer: str,
        evidence: Sequence[Dict[str, Any]],
    ) -> bool | None:
        q = re.sub(r"\s+", " ", str(query or "").lower()).strip()
        high = any(term in q for term in ("highest", "maximum", "largest"))
        low = any(term in q for term in ("lowest", "minimum", "smallest"))
        if not high and not low:
            return None

        if "price" in q:
            preferred_terms = ("price",)
        elif "revenue" in q:
            preferred_terms = ("revenue",)
        elif "units" in q or "unit sales" in q:
            preferred_terms = ("units", "sales")
        elif "share" in q:
            preferred_terms = ("share",)
        elif "score" in q or "csat" in q:
            preferred_terms = ("csat", "score")
        else:
            return None

        selected = None
        for item in evidence:
            if cls._source_kind(item) != "pdf":
                continue
            metadata = item.get("metadata") or {}
            if str(metadata.get("chunk_type", "")).lower() != "table":
                continue
            parsed = cls._pdf_parse_pipe_table(str(item.get("content", "") or ""))
            if parsed is None:
                continue
            headers, rows = parsed
            metric_index = cls._pdf_header_index(headers, preferred_terms)
            if metric_index is None:
                continue
            numeric_rows = [
                row
                for row in rows
                if len(row) > metric_index and cls._pdf_numeric_cell(row[metric_index]) is not None
            ]
            if len(numeric_rows) < 2:
                continue
            selected = (headers, numeric_rows, metric_index)
            break

        if selected is None:
            return None

        headers, rows, metric_index = selected
        direction = max if high else min
        winner = direction(
            rows,
            key=lambda row: cls._pdf_numeric_cell(row[metric_index]) or float("-inf"),
        )

        answer_lower = str(answer or "").lower()
        if winner[0].strip().lower() not in answer_lower:
            return False

        winner_value = cls._pdf_numeric_cell(winner[metric_index])
        if winner_value is None or not cls._pdf_number_present(answer, winner_value):
            return False

        # If the query asks for additional metrics for a row explicitly named
        # in the question, verify those exact cells too.
        named_row = None
        for row in rows:
            if row and row[0].strip().lower() in q:
                named_row = row
                break

        if named_row is not None:
            csat_index = cls._pdf_header_index(headers, ("csat", "customer satisfaction"))
            if (
                csat_index is not None
                and csat_index < len(named_row)
                and ("csat" in q or "customer satisfaction" in q)
            ):
                value = cls._pdf_numeric_cell(named_row[csat_index])
                if value is None or not cls._pdf_number_present(answer, value):
                    return False

            ev_index = cls._pdf_header_index(headers, ("ev", "share"), require_all=True)
            if (
                ev_index is not None
                and ev_index < len(named_row)
                and "ev" in q
                and "share" in q
            ):
                value = cls._pdf_numeric_cell(named_row[ev_index])
                if value is None or not cls._pdf_number_present(answer, value):
                    return False

        return True

    @classmethod
    def _pdf_explicit_list_supported(
        cls,
        *,
        query: str,
        answer: str,
        evidence: Sequence[Dict[str, Any]],
    ) -> bool | None:
        text_items = [
            item
            for item in evidence
            if (
                cls._source_kind(item) == "pdf"
                and str((item.get("metadata") or {}).get("chunk_type", "")).lower() == "text"
            )
        ]
        if not text_items:
            return None

        heading_terms = (
            "key market drivers",
            "key drivers",
            "market drivers",
            "key factors",
            "factors",
            "drivers",
        )
        focus_item = None
        matched_heading = ""
        for heading in heading_terms:
            matches = [
                item
                for item in text_items
                if heading in str(item.get("content", "") or "").lower()
            ]
            if matches:
                focus_item = max(
                    matches,
                    key=lambda item: float(item.get("relevance_score", 0.0) or 0.0),
                )
                matched_heading = heading
                break

        if focus_item is None:
            return None

        page_number = (focus_item.get("metadata") or {}).get("page_number")
        page_items = [
            item
            for item in text_items
            if (item.get("metadata") or {}).get("page_number") == page_number
        ]
        page_items.sort(
            key=lambda item: int((item.get("metadata") or {}).get("chunk_index") or 0)
        )

        combined = "\n".join(str(item.get("content", "") or "") for item in page_items)
        lower = combined.lower()
        start = lower.find(matched_heading)
        if start < 0:
            return None
        section = combined[start + len(matched_heading):]

        raw_parts = re.split(r"\s*[•●▪]\s*", section)
        if len(raw_parts) <= 1:
            return None

        bullets: list[str] = []
        for raw in raw_parts[1:]:
            cleaned = re.sub(r"\s+", " ", raw).strip(" -\t\r\n.;")
            if cleaned:
                bullets.append(cleaned)

        if len(bullets) < 3:
            return None

        answer_tokens = cls._pdf_content_tokens(answer)
        for bullet in bullets:
            bullet_tokens = cls._pdf_content_tokens(bullet)
            if not bullet_tokens:
                continue
            overlap = len(answer_tokens & bullet_tokens)
            required = 1 if len(bullet_tokens) <= 3 else 2
            if overlap < required:
                return False

        return True

    @staticmethod
    def _pdf_parse_pipe_table(
        content: str,
    ) -> tuple[list[str], list[list[str]]] | None:
        lines = [line.strip() for line in str(content or "").splitlines() if "|" in line]
        if len(lines) < 2:
            return None
        headers = [part.strip() for part in lines[0].split("|")]
        if len(headers) < 2:
            return None
        normalized_headers = [value.lower() for value in headers]
        rows: list[list[str]] = []
        for line in lines[1:]:
            row = [part.strip() for part in line.split("|")]
            if len(row) != len(headers):
                continue
            if [value.lower() for value in row] == normalized_headers:
                continue
            rows.append(row)
        return (headers, rows) if rows else None

    @staticmethod
    def _pdf_header_index(
        headers: Sequence[str],
        terms: Sequence[str],
        *,
        require_all: bool = False,
    ) -> int | None:
        normalized_terms = [re.sub(r"[^a-z0-9]+", " ", term.lower()).strip() for term in terms]
        for index, header in enumerate(headers):
            normalized = re.sub(r"[^a-z0-9]+", " ", str(header).lower()).strip()
            if require_all:
                if all(term in normalized for term in normalized_terms):
                    return index
            elif any(term in normalized for term in normalized_terms):
                return index
        return None

    @staticmethod
    def _pdf_numeric_cell(value: str) -> float | None:
        cleaned = re.sub(r"[^0-9.+-]", "", str(value or ""))
        if not cleaned or cleaned in {"+", "-", "."}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    @classmethod
    def _pdf_number_present(cls, text: str, value: float) -> bool:
        target = Decimal(str(value))
        for raw in cls._extract_numbers(str(text or "")):
            try:
                candidate = Decimal(raw)
            except InvalidOperation:
                continue
            if candidate == target:
                return True
        return False

    @staticmethod
    def _pdf_content_tokens(text: str) -> set[str]:
        stop = {
            "the", "and", "for", "from", "with", "that", "this", "were", "was",
            "are", "into", "toward", "where", "while", "their", "its", "over",
            "during", "continued", "supporting", "consistent", "existing", "model",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", str(text or "").lower())
            if token not in stop
        }

    # =========================================================
    # PDF-VISUAL-ONLY RELATION CHECK
    # =========================================================

    @classmethod
    def _pdf_visual_query_supported(
        cls,
        *,
        query: str,
        answer: str,
    ) -> bool:
        """
        Reject internally impossible PDF chart answers. The existing visual
        evidence whitelist still controls provenance; this adds only relation
        and requested-value consistency for explicit visual comparisons.
        """
        q = re.sub(r"\s+", " ", str(query or "").lower()).strip()
        a = re.sub(r"\s+", " ", str(answer or "").lower()).strip()

        if not a:
            return False

        inventory_relation = (
            "in production" in q
            and "current stock" in q
            and any(term in q for term in ("more", "greater", "higher", "less", "lower"))
        )

        if not inventory_relation:
            return True

        production = cls._labeled_visual_number(
            a,
            labels=("in production", "production"),
        )
        stock = cls._labeled_visual_number(
            a,
            labels=("current stock", "stock"),
        )

        if "give both values" in q and (production is None or stock is None):
            return False

        if production is None or stock is None:
            return False

        if any(term in q for term in ("more", "greater", "higher")):
            return production > stock

        if any(term in q for term in ("less", "lower")):
            return production < stock

        return True

    @staticmethod
    def _labeled_visual_number(
        text: str,
        *,
        labels: Sequence[str],
    ) -> float | None:
        """
        Read the number that is locally attached to a visual-series label.

        The previous clause-first parser could bind ``current stock`` to the
        production value in an answer such as::

            ... more units in production than current stock.
            In Production: approximately 310 units;
            Current Stock: approximately 265 units.

        because both sentences lived in one semicolon-delimited clause.  For
        PDF visual verification, prefer a number adjacent to the requested
        label and never cross sentence/semicolon boundaries.  The fallback
        clause scan is kept only after splitting sentence punctuation too.
        """
        source = str(text or "").lower()

        # Label -> number, constrained to the same punctuation-bounded phrase.
        # Handles ``Current Stock: approximately 265`` and
        # ``In Production is 310 units`` without borrowing a later metric.
        for label in labels:
            match = re.search(
                rf"{re.escape(label)}"
                rf"(?:\s+(?:is|are|was|were))?"
                rf"[^.;\n]{{0,32}}?"
                rf"([0-9][0-9,]*(?:\.[0-9]+)?)",
                source,
                re.IGNORECASE,
            )
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    pass

        # Number -> label in the same punctuation-bounded phrase.
        for label in labels:
            match = re.search(
                rf"([0-9][0-9,]*(?:\.[0-9]+)?)"
                rf"[^.;\n]{{0,24}}?(?:units?\s*)?"
                rf"(?:in\s+)?{re.escape(label)}",
                source,
                re.IGNORECASE,
            )
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    pass

        # Conservative fallback for short natural-language clauses.  Include
        # sentence punctuation so one metric cannot capture another sentence's
        # number.
        clauses = re.split(
            r"[.;\n]+|\b(?:and|while|whereas|versus|vs\.?)\b",
            source,
            flags=re.IGNORECASE,
        )
        for clause in clauses:
            if not any(label in clause for label in labels):
                continue
            numbers = re.findall(r"[0-9][0-9,]*(?:\.[0-9]+)?", clause)
            if len(numbers) == 1:
                try:
                    return float(numbers[0].replace(",", ""))
                except ValueError:
                    pass

        return None

    @classmethod
    def _query_complete(
        cls,
        *,
        query: str,
        answer: str,
    ) -> bool:
        """
        Reject answers that are grounded but omit an explicitly
        requested numeric part of a multi-part question.

        The check is intentionally conservative. It does not try to
        understand arbitrary natural language; it only protects clear
        quantitative requests such as ``how many``, ``how much``,
        multiple requested metrics, and ``both ... units and percentage``.
        """

        q = re.sub(
            r"\s+",
            " ",
            str(query or "").lower(),
        ).strip()

        if not q:
            return True

        required_numeric = len(
            re.findall(
                r"\bhow\s+(?:many|much)\b",
                q,
            )
        )

        what_match = re.search(
            r"\bwhat\s+(?:is|was|are|were)\b(?P<tail>.*)",
            q,
        )

        if what_match:
            tail = what_match.group(
                "tail"
            )

            metric_patterns = (
                r"\bcsat\s+score\b|\bscore\b",
                r"\bev\s+share\b|\bshare\b",
                r"\brevenue\b",
                r"\b(?:transaction\s+)?price\b",
                r"\bunits?(?:\s+sold)?\b",
                r"\bemployees?\b",
                r"\bamount\b",
                r"\brate\b",
            )

            metric_count = sum(
                1
                for pattern in metric_patterns
                if re.search(
                    pattern,
                    tail,
                )
            )

            # Multiple explicitly requested metrics require multiple
            # numeric values in the answer. A single metric is left to
            # the existing grounding checks so identification questions
            # such as "what was the highest revenue quarter" are not
            # incorrectly rejected.
            if metric_count >= 2:
                required_numeric = max(
                    required_numeric,
                    metric_count,
                )

        asks_percentage_terms = (
            any(
                phrase in q
                for phrase in (
                    "percentage terms",
                    "percentage change",
                    "percent change",
                    "what percentage",
                )
            )
            or bool(
                re.search(
                    r"\bhow\s+much\b.*\bpercent(?:age)?\b",
                    q,
                )
            )
        )

        asks_units_and_percentage = (
            "both" in q
            and bool(
                re.search(
                    r"\bunits?\b",
                    q,
                )
            )
            and bool(
                re.search(
                    r"\bpercent(?:age)?\b",
                    q,
                )
            )
        )

        if asks_units_and_percentage:
            required_numeric = max(
                required_numeric,
                2,
            )

        answer_numbers = (
            cls._extract_numbers(
                answer
            )
        )

        non_year_numbers = [
            value
            for value in answer_numbers
            if not cls._is_year_number_string(
                value
            )
        ]

        if (
            required_numeric > 0
            and len(non_year_numbers)
            < required_numeric
        ):
            return False

        if asks_percentage_terms:
            answer_lower = str(
                answer or ""
            ).lower()

            has_percentage = (
                "%" in answer_lower
                or " percent" in answer_lower
                or " percentage" in answer_lower
            )

            if not has_percentage:
                return False

        return True

    @staticmethod
    def _is_year_number_string(
        value: str,
    ) -> bool:
        try:
            decimal_value = Decimal(
                str(value)
            )
        except InvalidOperation:
            return False

        return VerificationAgent._is_year_value(
            decimal_value
        )

    # =========================================================
    # NORMALIZE TEXT
    # =========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize harmless formatting while preserving
        semantic content.
        """

        value = str(
            text
            or ""
        ).lower()

        value = value.replace(
            ",",
            "",
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    # =========================================================
    # FAILURE
    # =========================================================

    @classmethod
    def _fail(
        cls,
        reason: str,
        query: str,
    ) -> Dict[str, Any]:

        base_query = str(
            query
            or ""
        ).strip()

        instruction = (
            "Use only directly supported facts "
            "and cite the evidence."
        )

        if (
            instruction.lower()
            in base_query.lower()
        ):

            rewritten = (
                base_query
            )

        else:

            rewritten = (
                f"{base_query} "
                f"{instruction}"
            ).strip()

        log_event(
            logger,
            "verification",
            status="fail",
            reason=reason,
            evidence_count=0,
            used_evidence_count=0,
        )

        return {
            "status":
                "fail",

            "passed":
                False,

            "reason":
                reason,

            "grounding_overlap":
                0.0,

            "direct_support":
                False,

            "numeric_only_answer":
                False,

            "numeric_grounding_ok":
                False,

            "answer_numbers":
                [],

            "unsupported_numbers":
                [],

            "claims_grounded":
                False,

            "unsupported_claims":
                [],

            "used_evidence_ids":
                [],

            "conflict_acknowledged":
                False,

            "retry_required":
                True,

            "rewritten_query":
                rewritten,
        }

    # =========================================================
    # TOKENS
    # =========================================================

    @staticmethod
    def _tokens(
        text: str,
    ) -> set[str]:
        """
        Extract meaningful textual tokens.

        Numbers are deliberately handled independently by
        _extract_numbers().
        """

        stop_words = {
            "the",
            "and",
            "was",
            "were",
            "what",
            "from",
            "with",
            "that",
            "this",
            "only",
            "for",
            "are",
            "is",
            "at",
            "of",
            "in",
            "to",
        }

        return {
            token

            for token
            in re.findall(
                r"[a-zA-Z]{3,}",
                str(
                    text
                    or ""
                ).lower(),
            )

            if token
            not in stop_words
        }