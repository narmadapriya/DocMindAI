from __future__ import annotations

import re

from typing import Any, Dict, List, Sequence

from app.rag.agents.citation_whitelist import (
    CitationWhitelist,
)
from app.core.logging import get_logger, log_event

logger = get_logger(__name__)


class CitationAgent:
    """
    Phase 10 Citation Agent.

    Citations can only originate from
    the retrieved evidence whitelist.
    """

    def __init__(
        self,
        whitelist: CitationWhitelist
        | None = None,
    ):

        self.whitelist = (
            whitelist
            or CitationWhitelist()
        )

    def cite(
        self,
        answer: str,
        evidence: Sequence[
            Dict[str, Any]
        ],
        used_evidence_ids: Sequence[
            str
        ],
    ) -> Dict[str, Any]:

        allowed = (
            self.whitelist.build(
                evidence
            )
        )

        requested_ids = [
            str(value)
            for value
            in used_evidence_ids
            if str(value)
            in allowed
        ]

        # Preserve evidence order while removing duplicate IDs.
        requested_ids = list(
            dict.fromkeys(
                requested_ids
            )
        )

        # Guarantee a source for every
        # verified answer.
        if not requested_ids:

            requested_ids = list(
                allowed.keys()
            )[:1]

        selected = (
            self.whitelist.select(
                requested_ids,
                allowed,
            )
        )

        evidence_by_id = {
            str(
                item.get(
                    "chunk_id"
                )
            ): item
            for item in evidence
            if isinstance(
                item,
                dict,
            )
            and item.get(
                "chunk_id"
            ) is not None
        }

        selected = sorted(
            selected,
            key=lambda item: self._relevance_score(
                evidence_by_id.get(
                    str(
                        item.get(
                            "evidence_id",
                            "",
                        )
                    ),
                    {},
                )
            ),
            reverse=True,
        )

        selected = (
            self._remove_redundant_same_source_citations(
                answer=answer,
                selected=selected,
            )
        )

        # XLSX-only citation precision: when multiple sheets from the same
        # workbook were selected, keep sheets that actually support the final
        # answer's words/numbers. DOCX/CSV/PDF citations are untouched.
        selected = self._filter_weak_xlsx_citations(
            answer=answer,
            selected=selected,
        )

        citations: List[
            Dict[str, Any]
        ] = []

        for index, item in enumerate(
            selected,
            1,
        ):

            metadata = (
                item.get(
                    "metadata"
                )
                or {}
            )

            label = (
                self._format_source(
                    metadata
                )
            )

            citations.append(
                {
                    "citation_id":
                        f"S{index}",
                    "evidence_id":
                        item[
                            "evidence_id"
                        ],
                    "source":
                        label,
                    "text":
                        f"[Source: {label}]",
                }
            )

        citation_text = " ".join(
            citation["text"]
            for citation
            in citations
        )

        final_answer = (
            answer.strip()
        )

        if citation_text:

            final_answer = (
                f"{final_answer}\n\n"
                f"{citation_text}"
            )

        log_event(
            logger,
            "citation",
            citation_count=len(citations),
            evidence_count=len(evidence),
            status=("success" if citations else "no_citations"),
        )

        return {
            "answer": final_answer,
            "citations": citations,
            "citation_count":
                len(citations),
            "status": (
                "success"
                if citations
                else "no_citations"
            ),
        }


    @staticmethod
    def _relevance_score(
        item: Dict[str, Any],
    ) -> float:
        try:
            return float(
                item.get(
                    "relevance_score",
                    0.0,
                )
                or 0.0
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @classmethod
    def _remove_redundant_same_source_citations(
        cls,
        *,
        answer: str,
        selected: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove only clearly redundant citations from the same source.

        Different files are never collapsed. Different locations in the
        same file are also retained when they support different parts of
        the final answer. This keeps multi-part citation coverage while
        avoiding duplicate related pages that support exactly the same
        answer content.
        """

        kept: List[Dict[str, Any]] = []
        support_by_source: Dict[
            str,
            List[frozenset[str]],
        ] = {}

        for item in selected:
            metadata = (
                item.get(
                    "metadata"
                )
                or {}
            )

            source_key = str(
                metadata.get(
                    "filename"
                )
                or metadata.get(
                    "file_name"
                )
                or metadata.get(
                    "document_id"
                )
                or "unknown source"
            )

            signature = (
                cls._answer_support_signature(
                    answer=answer,
                    content=str(
                        item.get(
                            "content",
                            "",
                        )
                    ),
                )
            )

            previous_signatures = (
                support_by_source.get(
                    source_key,
                    [],
                )
            )

            redundant = bool(
                signature
                and any(
                    signature.issubset(
                        previous
                    )
                    for previous
                    in previous_signatures
                    if previous
                )
            )

            if redundant:
                continue

            kept.append(item)
            support_by_source.setdefault(
                source_key,
                [],
            ).append(signature)

        return kept

    @staticmethod
    def _answer_support_signature(
        *,
        answer: str,
        content: str,
    ) -> frozenset[str]:
        answer_lower = str(
            answer or ""
        ).lower()
        content_lower = str(
            content or ""
        ).lower()

        stop_words = {
            "the",
            "and",
            "was",
            "were",
            "with",
            "from",
            "that",
            "this",
            "for",
            "are",
            "is",
            "in",
            "to",
            "of",
        }

        answer_tokens = {
            token
            for token in re.findall(
                r"[a-zA-Z][a-zA-Z0-9-]{2,}",
                answer_lower,
            )
            if token not in stop_words
        }

        content_tokens = set(
            re.findall(
                r"[a-zA-Z][a-zA-Z0-9-]{2,}",
                content_lower,
            )
        )

        signature = {
            f"t:{token}"
            for token in (
                answer_tokens
                & content_tokens
            )
        }

        def numbers(text: str) -> set[str]:
            return {
                value
                .replace(",", "")
                .replace("%", "")
                for value in re.findall(
                    r"(?<!\w)"
                    r"[-+]?"
                    r"(?:\d{1,3}(?:,\d{3})+|\d+)"
                    r"(?:\.\d+)?%?",
                    text,
                )
            }

        signature.update(
            f"n:{value}"
            for value in (
                numbers(answer_lower)
                & numbers(content_lower)
            )
        )

        return frozenset(
            signature
        )

    @classmethod
    def _filter_weak_xlsx_citations(
        cls,
        *,
        answer: str,
        selected: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if len(selected) <= 1:
            return list(selected)

        def is_xlsx(item: Dict[str, Any]) -> bool:
            metadata = item.get("metadata") or {}
            filename = str(
                metadata.get("filename")
                or metadata.get("file_name")
                or ""
            ).lower()
            return filename.endswith((".xlsx", ".xls"))

        if not any(is_xlsx(item) for item in selected):
            return list(selected)

        xlsx_items = [item for item in selected if is_xlsx(item)]
        other_items = [item for item in selected if not is_xlsx(item)]

        scored: list[tuple[int, Dict[str, Any]]] = []
        for item in xlsx_items:
            signature = cls._answer_support_signature(
                answer=answer,
                content=str(item.get("content", "") or ""),
            )
            scored.append((len(signature), item))

        best = max((score for score, _ in scored), default=0)
        # Keep any workbook location that contributes at least one answer fact;
        # if all signatures are empty, preserve the original citations.
        kept_xlsx = (
            [item for score, item in scored if score > 0 and score >= max(1, best // 2)]
            if best > 0
            else xlsx_items
        )

        return other_items + kept_xlsx

    @staticmethod
    def _format_source(
        metadata: Dict[str, Any],
    ) -> str:

        filename = (
            metadata.get(
                "filename"
            )
            or metadata.get(
                "file_name"
            )
            or metadata.get(
                "document_id"
            )
            or "unknown source"
        )

        page = metadata.get(
            "page_number"
        )

        sheet = metadata.get(
            "sheet_name"
        )

        table_id = metadata.get(
            "table_id"
        )

        source_location = (
            metadata.get(
                "source_location"
            )
        )

        if table_id:
            return (
                f"{filename}, "
                f"Table {table_id}"
            )

        if sheet:
            return (
                f"{filename}, "
                f"Sheet: {sheet}"
            )

        if page is not None:
            return (
                f"{filename}, "
                f"Page {page}"
            )

        if source_location:
            return (
                f"{filename}, "
                f"{source_location}"
            )

        return str(
            filename
        )