from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence


class ChatAgent:
    """
    Phase 11 user-facing document QA agent.

    Responsibilities:
        - document-aware QA
        - multi-document QA
        - conversation-aware follow-ups
        - multimodal context preparation
        - delegation to the Phase 10 reasoning layer

    This agent does NOT perform vector retrieval itself.
    Retrieval remains owned by Phase 9.

    Phase-15 presentation hardening:
        - keeps the existing grounded answer and citation flow unchanged
        - renders the final Chat & Ask answer in the DocMindAI standard:
          ANSWER -> supporting explanation -> optional KEY FINDINGS
        - keeps sources separate in the existing citations field
    """

    _PRESENTATION_HEADINGS = {
        "answer",
        "explanation",
        "key points",
        "key findings",
        "conclusion",
        "comparison",
        "key observations",
        "values used",
        "calculation",
        "relevant rows / columns",
        "interpretation",
        "visual evidence",
        "important values / trends",
        "key takeaways",
    }

    _KEY_SECTION_HEADINGS = {
        "key points",
        "key findings",
        "key observations",
        "key takeaways",
    }

    _SUPPORT_SECTION_HEADINGS = {
        "explanation",
        "comparison",
        "values used",
        "calculation",
        "relevant rows / columns",
        "interpretation",
        "visual evidence",
        "important values / trends",
    }

    _COMPLEX_QUERY_TERMS = (
        "explain",
        "describe",
        "detail",
        "compare",
        "comparison",
        "difference",
        "calculate",
        "calculation",
        "percentage",
        "percent",
        "increase",
        "decrease",
        "trend",
        "chart",
        "graph",
        "image",
        "figure",
        "diagram",
        "table",
        "spreadsheet",
        "worksheet",
        "types",
        "kinds",
        "categories",
        "components",
        "steps",
        "list",
        "summarize",
        "summary",
        "overview",
    )

    def __init__(
        self,
        reasoning_agent=None,
        citation_agent=None,
    ):
        self.reasoning_agent = reasoning_agent
        self.citation_agent = citation_agent

    # =========================================================
    # PUBLIC API
    # =========================================================

    def answer(
        self,
        *,
        query: str,
        evidence: Sequence[Dict[str, Any]],
        history: Optional[Sequence[Dict[str, Any]]] = None,
        validation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        query = str(query or "").strip()

        if not query:
            raise ValueError("Chat query cannot be empty.")

        evidence = list(evidence or [])
        history = list(history or [])

        contextual_query = self._build_contextual_query(
            query=query,
            history=history,
        )

        if self.reasoning_agent is None:
            answer = self._fallback_answer(
                query=contextual_query,
                evidence=evidence,
            )
        else:
            result = self.reasoning_agent.reason(
                contextual_query,
                evidence,
                validation=validation or {
                    "valid_evidence": evidence
                },
            )

            if isinstance(result, dict):
                answer = (
                    result.get("answer")
                    or result.get("content")
                    or result.get("response")
                    or str(result)
                )
            else:
                answer = str(result)

        citations = self._build_citations(evidence)

        if self.citation_agent is not None:
            try:
                citation_result = self.citation_agent.cite(
                    answer=answer,
                    evidence=evidence,
                )

                if isinstance(citation_result, dict):
                    citations = (
                        citation_result.get("citations")
                        or citations
                    )
                    answer = (
                        citation_result.get("answer")
                        or answer
                    )
            except (AttributeError, TypeError):
                pass

        # Presentation only. The reasoning result itself is not regenerated,
        # and citations remain in the existing dedicated response field.
        answer = self._format_chat_answer(
            query=query,
            answer=answer,
        )

        return {
            "answer": answer.strip(),
            "citations": citations,
            "evidence_count": len(evidence),
            "conversation_context_used": bool(history),
            "document_count": self._document_count(evidence),
        }

    # =========================================================
    # FOLLOW-UP CONTEXT
    # =========================================================

    @staticmethod
    def _build_contextual_query(
        *,
        query: str,
        history: Sequence[Dict[str, Any]],
        max_turns: int = 6,
    ) -> str:

        if not history:
            return query

        recent = list(history)[-max_turns:]

        lines = [
            "Conversation history:",
        ]

        for item in recent:

            role = str(
                item.get("role", "user")
            ).upper()

            content = str(
                item.get("content", "")
            ).strip()

            if not content:
                continue

            lines.append(
                f"{role}: {content}"
            )

        lines.extend(
            [
                "",
                "Current user question:",
                query,
                "",
                (
                    "Resolve references such as "
                    "'it', 'that document', 'the previous "
                    "number', or 'this value' using the "
                    "conversation history when possible."
                ),
            ]
        )

        return "\n".join(lines)

    # =========================================================
    # CHAT PRESENTATION
    # =========================================================

    @classmethod
    def _format_chat_answer(
        cls,
        *,
        query: str,
        answer: str,
    ) -> str:
        """
        Reorganize the already-grounded answer for the Chat & Ask UI.

        This method does not create new document facts. It removes legacy
        presentation labels/citation markers, preserves the existing answer
        content, and renders a stable Markdown-safe structure that ReactMarkdown
        displays as separate blocks instead of collapsing into one paragraph.
        """
        clean = cls._strip_inline_sources(answer)
        if not clean:
            return clean

        sections = cls._parse_presentation_sections(clean)

        answer_section = sections.get("answer", [])
        direct = cls._join_lines(answer_section)

        if not direct:
            direct = cls._first_paragraph(clean)

        if not direct:
            direct = clean

        support_blocks: List[str] = []
        for heading in (
            "explanation",
            "comparison",
            "values used",
            "calculation",
            "relevant rows / columns",
            "interpretation",
            "visual evidence",
            "important values / trends",
        ):
            body = cls._join_lines(sections.get(heading, []))
            if body:
                support_blocks.append(body)

        key_findings: List[str] = []
        for heading in (
            "key findings",
            "key points",
            "key observations",
            "key takeaways",
        ):
            key_findings.extend(
                cls._extract_list_items(
                    sections.get(heading, [])
                )
            )

        # When the older reasoning formatter did not create explicit sections,
        # preserve the first paragraph as the direct answer and use the
        # remaining grounded paragraphs as supporting explanation.
        if not sections:
            paragraphs = cls._paragraphs(clean)
            if paragraphs:
                direct = paragraphs[0]
                support_blocks.extend(paragraphs[1:])

        # Older list/detail answers can place numbered items in Explanation.
        # Use those existing items as findings rather than generating new facts.
        if not key_findings:
            for block in support_blocks:
                key_findings.extend(
                    cls._extract_list_items(block.splitlines())
                )

        key_findings = cls._dedupe_findings(key_findings)

        include_key_findings = cls._should_show_key_findings(
            query=query,
            findings=key_findings,
            support_blocks=support_blocks,
        )

        if include_key_findings and not key_findings:
            key_findings = cls._findings_from_support(support_blocks)

        key_findings = cls._dedupe_findings(key_findings)[:4]

        # Do not repeat the exact direct answer in the explanation.
        compact_direct = cls._compact(direct)
        filtered_support: List[str] = []
        for block in support_blocks:
            value = cls._clean_support_block(block)
            if not value:
                continue
            if cls._compact(value) == compact_direct:
                continue
            filtered_support.append(value)

        parts = [
            "ANSWER",
            "────────────────────────────",
            direct.strip(),
        ]

        if filtered_support:
            parts.append("\n\n".join(filtered_support))

        if include_key_findings and key_findings:
            parts.append(
                "KEY FINDINGS\n\n"
                + "\n".join(
                    f"- {finding}"
                    for finding in key_findings
                )
            )

        return "\n\n".join(
            part.strip()
            for part in parts
            if str(part or "").strip()
        ).strip()

    @classmethod
    def _parse_presentation_sections(
        cls,
        value: str,
    ) -> Dict[str, List[str]]:
        sections: Dict[str, List[str]] = {}
        current: str | None = None
        recognized = False

        for raw_line in str(value or "").replace("\r", "\n").splitlines():
            line = raw_line.strip()
            normalized = cls._normalize_heading(line)

            if normalized in cls._PRESENTATION_HEADINGS:
                current = normalized
                sections.setdefault(current, [])
                recognized = True
                continue

            if current is not None:
                sections[current].append(raw_line)

        return sections if recognized else {}

    @staticmethod
    def _normalize_heading(value: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"^[#*_`\s]+|[#*_`\s:]+$", "", text)
        return re.sub(r"\s+", " ", text).strip().lower()

    @classmethod
    def _strip_inline_sources(cls, value: str) -> str:
        text = str(value or "")
        text = re.sub(
            r"\s*\[\s*Source\s*:\s*[^\]]+\]",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"(?is)(?:^|\n)\s*Sources\s*/?\s*Citations\s*:?.*$",
            "",
            text,
        )
        text = re.sub(
            r"(?is)(?:^|\n)\s*Sources\s*:?.*$",
            "",
            text,
        )
        return re.sub(r"[ \t]+\n", "\n", text).strip()

    @staticmethod
    def _paragraphs(value: str) -> List[str]:
        return [
            re.sub(r"\s+", " ", part).strip()
            for part in re.split(r"\n\s*\n+", str(value or ""))
            if re.sub(r"\s+", " ", part).strip()
        ]

    @classmethod
    def _first_paragraph(cls, value: str) -> str:
        paragraphs = cls._paragraphs(value)
        return paragraphs[0] if paragraphs else ""

    @staticmethod
    def _join_lines(lines: Sequence[str]) -> str:
        cleaned = [
            line.strip()
            for line in lines
            if str(line or "").strip()
        ]
        return "\n".join(cleaned).strip()

    @classmethod
    def _extract_list_items(
        cls,
        lines: Sequence[str],
    ) -> List[str]:
        items: List[str] = []
        for raw in lines:
            line = str(raw or "").strip()
            if not line:
                continue

            match = re.match(
                r"^(?:[-*•]\s+|\d{1,2}[.)]\s+)(.+)$",
                line,
            )
            if match:
                item = match.group(1).strip()
                if item:
                    items.append(item)

        return items

    @classmethod
    def _findings_from_support(
        cls,
        support_blocks: Sequence[str],
    ) -> List[str]:
        findings: List[str] = []
        for block in support_blocks:
            normalized = re.sub(r"\s+", " ", str(block or "")).strip()
            if not normalized:
                continue

            sentences = [
                sentence.strip()
                for sentence in re.split(
                    r"(?<=[.!?])\s+",
                    normalized,
                )
                if sentence.strip()
            ]

            findings.extend(sentences[:2])
            if len(findings) >= 4:
                break

        return findings[:4]

    @classmethod
    def _should_show_key_findings(
        cls,
        *,
        query: str,
        findings: Sequence[str],
        support_blocks: Sequence[str],
    ) -> bool:
        q = re.sub(r"\s+", " ", str(query or "").lower()).strip()

        if findings:
            return True

        if any(term in q for term in cls._COMPLEX_QUERY_TERMS):
            return bool(support_blocks)

        # Very simple factual questions intentionally remain concise.
        return False

    @classmethod
    def _dedupe_findings(
        cls,
        findings: Sequence[str],
    ) -> List[str]:
        output: List[str] = []
        seen = set()

        for value in findings:
            cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" -•\t")
            key = cls._compact(cleaned)
            if not cleaned or not key or key in seen:
                continue
            seen.add(key)
            output.append(cleaned)

        return output

    @staticmethod
    def _clean_support_block(value: str) -> str:
        text = str(value or "").strip()
        text = re.sub(
            r"^(?:Conclusion\s*)",
            "",
            text,
            flags=re.I,
        )
        return text.strip()

    @staticmethod
    def _compact(value: str) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            " ",
            str(value or "").lower(),
        ).strip()

    # =========================================================
    # CITATIONS
    # =========================================================

    @staticmethod
    def _build_citations(
        evidence: Sequence[Dict[str, Any]],
    ) -> List[str]:

        citations: List[str] = []
        seen = set()

        for item in evidence:

            metadata = dict(
                item.get("metadata") or {}
            )

            filename = (
                metadata.get("filename")
                or metadata.get("file_name")
                or metadata.get("source")
            )

            if not filename:
                continue

            page = (
                metadata.get("page_number")
                if metadata.get("page_number") is not None
                else metadata.get("page")
            )

            section = (
                metadata.get("section")
                or metadata.get("section_name")
                or metadata.get("heading")
            )

            sheet = (
                metadata.get("sheet_name")
                or metadata.get("sheet")
            )

            table = (
                metadata.get("table_name")
                or metadata.get("table_id")
            )

            location: List[str] = []

            if page is not None:
                location.append(f"Page {page}")

            if section:
                location.append(f"Section: {section}")

            if sheet:
                location.append(f"Sheet: {sheet}")

            if table:
                location.append(f"Table: {table}")

            if location:
                citation = (
                    f"[Source: {filename}, "
                    + ", ".join(location)
                    + "]"
                )
            else:
                citation = (
                    f"[Source: {filename}]"
                )

            # Preserve the baseline citation-card count. Before the display
            # metadata enrichment, citations on the same file/page (or sheet)
            # collapsed to one card. Keep that identity rule while showing the
            # richer section/table metadata from the first supporting chunk.
            if page is not None:
                identity = ("page", str(filename), str(page))
            elif sheet:
                identity = ("sheet", str(filename), str(sheet))
            elif table:
                identity = ("table", str(filename), str(table))
            else:
                identity = ("file", str(filename))

            if identity not in seen:
                seen.add(identity)
                citations.append(citation)

        return citations

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _document_count(
        evidence: Sequence[Dict[str, Any]],
    ) -> int:

        ids = set()

        for item in evidence:

            metadata = dict(
                item.get("metadata") or {}
            )

            document_id = (
                metadata.get("document_id")
                or metadata.get("filename")
            )

            if document_id:
                ids.add(document_id)

        return len(ids)

    @staticmethod
    def _fallback_answer(
        *,
        query: str,
        evidence: Sequence[Dict[str, Any]],
    ) -> str:

        if not evidence:
            return (
                "I could not find supporting evidence "
                "in the selected documents."
            )

        contents = []

        for item in evidence:

            content = str(
                item.get("content")
                or item.get("text")
                or ""
            ).strip()

            if content:
                contents.append(content)

        if not contents:
            return (
                "The selected documents contain no "
                "usable evidence for this question."
            )

        return (
            "Based on the retrieved document evidence:\n\n"
            + "\n\n".join(contents[:5])
        )
