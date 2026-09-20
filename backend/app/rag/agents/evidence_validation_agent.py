from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


class EvidenceValidationAgent:
    """
    Phase 10 Evidence Validation Agent.

    Validates retrieved evidence before reasoning.

    Checks:

        - relevance
        - completeness
        - source availability
        - conflicting evidence
    """

    def __init__(
        self,
        min_relevance: float = 0.20,
    ):
        self.min_relevance = float(
            min_relevance
        )

    # =========================================================
    # PUBLIC
    # =========================================================

    def validate(
        self,
        query: str,
        evidence: Sequence[
            Dict[str, Any]
        ],
        *,
        min_relevance: float | None = None,
    ) -> Dict[str, Any]:

        threshold = (
            self.min_relevance
            if min_relevance is None
            else float(min_relevance)
        )

        items = [
            item
            for item in evidence
            if isinstance(item, dict)
        ]

        relevance_ok = bool(items) and all(
            0.0
            <= self._safe_score(item)
            <= 1.0
            and self._safe_score(item)
            >= threshold
            for item in items
        )

        source_ok = bool(items) and all(
            self._source_available(item)
            for item in items
        )

        completeness_ok = self._complete(
            query,
            items,
        )

        conflicts = self._find_conflicts(
            items
        )

        valid_evidence = [
            item
            for item in items
            if (
                self._safe_score(item)
                >= threshold
                and self._source_available(item)
            )
        ]

        status = (
            "pass"
            if valid_evidence
            and completeness_ok
            else "fail"
        )

        issues: List[str] = []

        if not items:
            issues.append(
                "no_evidence"
            )

        if items and not relevance_ok:
            issues.append(
                "low_relevance"
            )

        if items and not source_ok:
            issues.append(
                "missing_source"
            )

        if items and not completeness_ok:
            issues.append(
                "incomplete_evidence"
            )

        if conflicts:
            issues.append(
                "conflicting_evidence"
            )

        return {
            "status": status,
            "valid_evidence": valid_evidence,
            "evidence_count": len(items),
            "valid_evidence_count": len(
                valid_evidence
            ),
            "relevance_ok": relevance_ok,
            "completeness_ok": completeness_ok,
            "source_availability_ok": source_ok,
            "conflicting_evidence": bool(
                conflicts
            ),
            "conflicts": conflicts,
            "issues": issues,
        }

    # =========================================================
    # RELEVANCE
    # =========================================================

    @staticmethod
    def _safe_score(
        evidence: Dict[str, Any],
    ) -> float:

        try:
            return max(
                0.0,
                min(
                    1.0,
                    float(
                        evidence.get(
                            "relevance_score",
                            0.0,
                        )
                    ),
                ),
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    # =========================================================
    # SOURCE AVAILABILITY
    # =========================================================

    @staticmethod
    def _source_available(
        evidence: Dict[str, Any],
    ) -> bool:

        metadata = (
            evidence.get(
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

        document_id = metadata.get(
            "document_id"
        )

        return bool(
            filename
            or document_id
        )

    # =========================================================
    # COMPLETENESS
    # =========================================================

    @staticmethod
    def _tokens(
        text: str,
    ) -> set[str]:

        stop_words = {
            "what",
            "was",
            "were",
            "the",
            "and",
            "for",
            "from",
            "does",
            "show",
            "about",
            "with",
            "this",
            "that",
            "tell",
            "please",
        }

        return {
            token
            for token in re.findall(
                r"[a-zA-Z0-9]{3,}",
                text.lower(),
            )
            if token not in stop_words
        }

    def _complete(
        self,
        query: str,
        evidence: Sequence[
            Dict[str, Any]
        ],
    ) -> bool:

        if not evidence:
            return False

        query_tokens = self._tokens(
            query
        )

        if not query_tokens:
            return True

        combined = " ".join(
            str(
                item.get(
                    "content",
                    "",
                )
            )
            for item in evidence
        )

        evidence_tokens = self._tokens(
            combined
        )

        overlap = (
            query_tokens
            & evidence_tokens
        )

        return bool(overlap) or any(
            self._safe_score(item)
            >= 0.75
            for item in evidence
        )

    # =========================================================
    # CONFLICT DETECTION
    # =========================================================

    def _find_conflicts(
        self,
        evidence: Sequence[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Detect genuine contradictory narrative claims.

        Structured evidence (DOCX/XLSX/CSV tables and row-record
        chunks) legitimately contains many numeric values for the same
        year and topic. The previous implementation grouped every
        number in such a block under (year, topic), so quarterly,
        monthly, unit, price and revenue values were incorrectly marked
        as conflicts.

        A value is conflict-comparable only when one sentence or line
        expresses an unambiguous topic/year numeric claim:

            - exactly one year
            - exactly one non-year numeric value

        This keeps genuine contradiction detection while preventing
        structured table/row values from being treated as conflicting
        versions of one value.
        """

        groups: Dict[
            Tuple[str, str],
            List[
                Tuple[
                    str,
                    float,
                    str,
                ]
            ],
        ] = defaultdict(list)

        for item in evidence:

            text = str(
                item.get(
                    "content",
                    "",
                )
            )

            metadata = (
                item.get(
                    "metadata"
                )
                or {}
            )

            source = str(
                metadata.get(
                    "filename"
                )
                or metadata.get(
                    "document_id"
                )
                or "unknown"
            )

            # DOCX/XLSX/CSV evidence frequently represents
            # independent records on separate lines.
            #
            # Split lines before numeric comparison so a complete
            # table or record block is never interpreted as one claim.
            claim_units = re.split(
                r"(?:\r?\n)+|(?<=[.!?])\s+",
                text,
            )

            for claim in claim_units:

                claim = claim.strip()

                if not claim:
                    continue

                topic = self._topic(
                    claim
                )

                if topic == "numeric_claim":
                    continue

                years = list(
                    dict.fromkeys(
                        re.findall(
                            r"(?<!\d)(20\d{2})(?!\d)",
                            claim,
                        )
                    )
                )

                # Claims referring to several years describe a
                # comparison/trend and must not be treated as two
                # contradictory versions of one value.
                if len(years) != 1:
                    continue

                values = self._extract_numbers(
                    claim
                )

                non_year_values = [
                    value
                    for value in values
                    if value[0] not in years
                ]

                # Example structured row:
                #
                # Year: 2025; Units Sold: 1050;
                # Price: 36800; Revenue: 38640000
                #
                # It contains several independent metrics. They are
                # not contradictory values for "revenue".
                #
                # Only an unambiguous single numeric claim should
                # participate in cross-source conflict detection.
                if len(non_year_values) != 1:
                    continue

                _, numeric_value = (
                    non_year_values[0]
                )

                groups[
                    (
                        years[0],
                        topic,
                    )
                ].append(
                    (
                        source,
                        numeric_value,
                        claim,
                    )
                )

        conflicts: List[
            Dict[str, Any]
        ] = []

        for (
            year,
            topic,
        ), observations in groups.items():

            # Repeated chunks/citations containing the exact same
            # statement must not manufacture a conflict.
            unique_observations: List[
                Tuple[str, float, str]
            ] = []

            seen = set()

            for observation in observations:

                marker = (
                    observation[0],
                    round(
                        observation[1],
                        8,
                    ),
                    observation[2],
                )

                if marker in seen:
                    continue

                seen.add(
                    marker
                )

                unique_observations.append(
                    observation
                )

            unique_values = {
                round(
                    observation[1],
                    8,
                )
                for observation
                in unique_observations
            }

            if len(unique_values) <= 1:
                continue

            conflicts.append(
                {
                    "year": year,
                    "topic": topic,
                    "values": [
                        {
                            "source": observation[0],
                            "value": observation[1],
                        }
                        for observation
                        in unique_observations
                    ],
                }
            )

        return conflicts

    @staticmethod
    def _extract_numbers(
        text: str,
    ) -> List[
        Tuple[str, float]
    ]:
        """
        Extract complete numeric values without splitting
        comma-formatted numbers.

        Example:

            $181,100,000

        is extracted as one numeric value instead of:

            181
            100
            000
        """

        matches = re.findall(
            r"(?<!\w)"
            r"[-+]?"
            r"(?:"
            r"\d{1,3}(?:,\d{3})+"
            r"|"
            r"\d+"
            r")"
            r"(?:\.\d+)?"
            r"%?",
            str(
                text
                or ""
            ),
        )

        output: List[
            Tuple[str, float]
        ] = []

        for raw_value in matches:

            normalized = (
                raw_value
                .replace(
                    ",",
                    "",
                )
                .replace(
                    "%",
                    "",
                )
                .strip()
            )

            if not normalized:
                continue

            try:
                numeric_value = float(
                    normalized
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            output.append(
                (
                    normalized,
                    numeric_value,
                )
            )

        return output

    @staticmethod
    def _topic(
        text: str,
    ) -> str:

        lower = text.lower()

        for topic in (
            "revenue",
            "sales",
            "profit",
            "income",
            "amount",
        ):
            if topic in lower:
                return topic

        return "numeric_claim"