from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


class SummaryAgent:
    """
    DocMindAI summary-generation agent.

    The public contract is unchanged.  File Summary and Executive Summary are
    deterministic, evidence-only presentation layers over the evidence already
    selected by SummaryService.  No retrieval, API, authentication, database,
    or shared RAG behaviour is changed here.

    - File Summary: 200-250 words.
    - Executive Summary: exactly five concise lines.
    - Sources are returned only through the existing citations field.
    """

    FILE_SUMMARY_MIN_WORDS = 200
    FILE_SUMMARY_MAX_WORDS = 250
    FILE_SUMMARY_TARGET_WORDS = 225

    EXECUTIVE_SUMMARY_LINES = 5

    SECTION_NUM_PREDICT = 220
    SECTION_NUM_CTX = 3072

    _NOISE_LINE_PATTERNS = (
        r"powered by great learning",
        r"proprietary content",
        r"all rights reserved",
        r"unauthori[sz]ed use or distribution prohibited",
        r"^use or distribution prohibited\.?$",
        r"this file is meant for personal use",
        r"sharing or publishing the contents",
        r"^(?=.*\d)[A-Z0-9]{8,}$",
        r"^[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}$",
        r"test-use note:",
        r"a good file summary should",
        r"a good executive summary should",
        r"rag summarization test case",
        r"fictional test document for docmindai evaluation",
        r"fictional test document for file summary",
        r"table of contents",
        r"program:\s*[^.]+specialization:",
        r"course code:",
        r"this file is meant for personal use",
        r"image content extracted from document",
        r"chart/graph extracted from document",
        r"chart extracted from document",
    )

    _PURPOSE_TERMS = (
        "overview", "objective", "purpose", "introduce",
        "learning outcome", "focus", "covers", "describes", "explains",
        "data communication is", "network devices or network hardware",
    )
    _METRIC_TERMS = (
        "revenue", "margin", "units", "deliver", "employee", "percent",
        "%", "score", "subscriptions", "hours", "capacity", "growth",
        "elements", "types", "categories", "components", "modes",
    )
    _RISK_TERMS = (
        "risk", "disadvantage", "drawback", "failure", "weakness",
        "constraint", "delay", "issue", "problem", "limitation",
    )
    _OUTLOOK_TERMS = (
        "outlook", "target", "plans to", "plan to", "priority",
        "expects", "intends", "future", "fy2027", "fy2026",
    )
    _CONCLUSION_TERMS = (
        "conclusion", "overall assessment", "overall",
        "in summary", "in conclusion",
    )
    _VISUAL_TERMS = (
        "figure", "fig.", "chart", "graph", "diagram", "table",
    )
    _SALIENT_TERMS = (
        "revenue", "growth", "highest", "largest", "fastest", "improved",
        "network", "protocol", "topology", "device", "data communication",
        "customer", "product", "operations", "risk", "target", "outlook",
    )

    _HIGH_LEVEL_SECTION_TERMS = (
        "overview", "introduction", "company overview", "financial",
        "sales performance", "product", "customer", "operations",
        "supply chain", "risk", "regional", "outlook", "overall assessment",
        "components of data communication", "data representation", "data flow",
        "network devices", "types of connection", "topology",
        "categories of networks", "applications of networks", "history of network",
        "protocols", "standards", "conclusion",
    )
    _LOW_LEVEL_SECTION_TERMS = (
        "pre-unit preparatory", "post-unit reading", "glossary", "references",
        "bibliography", "learning outcomes", "objectives", "text", "numbers",
        "images", "audio", "video", "example", "ex. ", "binary to",
        "decimal to", "octal to", "hexadecimal to",
    )
    _OPPORTUNITY_TERMS = (
        "benefit", "advantage", "application", "used for", "enables",
        "helps", "improve", "growth", "opportunity", "expansion", "target",
        "plan", "priority",
    )

    def __init__(self, reasoning_agent=None):
        self.reasoning_agent = reasoning_agent

    # =========================================================
    # PUBLIC API
    # =========================================================

    def summarize(
        self,
        *,
        evidence: Sequence[Dict[str, Any]],
        scope: str = "document",
        title: str = "",
    ) -> Dict[str, Any]:
        clean_evidence = [
            item for item in list(evidence or [])
            if isinstance(item, dict) and self._clean_content(self._content(item))
        ]

        if not clean_evidence:
            return {
                "summary": "No supporting evidence was found for the requested summary.",
                "citations": [],
                "scope": scope,
                "evidence_count": 0,
                "document_count": 0,
            }

        normalized_scope = str(scope or "document").strip().lower()

        if normalized_scope == "document":
            summary, used_ids = self._file_summary(clean_evidence)
        elif normalized_scope == "executive":
            summary, used_ids = self._executive_summary(clean_evidence)
        else:
            summary = self._existing_scope_summary(
                evidence=clean_evidence,
                scope=normalized_scope,
                title=title,
            )
            used_ids = [
                str(item.get("chunk_id"))
                for item in clean_evidence
                if item.get("chunk_id") is not None
            ]

        summary = self._strip_source_markers(str(summary or "").strip())
        used = self._evidence_by_ids(clean_evidence, used_ids)
        if not used:
            used = clean_evidence[:3]

        return {
            "summary": summary,
            "citations": self._citations(used),
            "scope": scope,
            "evidence_count": len(clean_evidence),
            "document_count": self._document_count(clean_evidence),
        }

    # =========================================================
    # FILE SUMMARY: STRICT 200-250 WORDS
    # =========================================================

    def _file_summary(
        self,
        evidence: Sequence[Dict[str, Any]],
    ) -> tuple[str, list[str]]:
        facts = self._facts(evidence)
        if not facts:
            text = self._safe_extract(evidence, max_words=self.FILE_SUMMARY_MAX_WORDS)
            return text, self._all_ids(evidence[:4])

        purpose_fact = self._pick_summary_purpose(facts)
        representatives = self._high_level_section_representatives(facts)
        if not representatives:
            representatives = self._section_representatives(facts)
        representatives = [
            fact for fact in representatives
            if not self._is_low_value_summary_fact(fact)
        ]

        # One source-grounded representative per high-level section, in source order.
        major: list[dict[str, Any]] = []
        major_seen: set[str] = set()
        for fact in sorted(representatives, key=lambda f: f["position"]):
            if purpose_fact and fact["key"] == purpose_fact["key"]:
                continue
            section_key = self._compact(str(fact.get("section") or ""))
            if section_key in major_seen:
                continue
            major_seen.add(section_key)
            major.append(fact)

        metric_candidates = [
            fact for fact in self._rank_facts(facts, predicate=lambda f: self._is_metric_fact(f))
            if not self._is_low_value_summary_fact(fact)
            and not self._is_conversion_example_fact(fact)
        ]
        evidence_candidates = [
            fact for fact in self._rank_facts(
                facts,
                predicate=lambda f: (
                    f["source_kind"] in {"table", "chart", "image"}
                    or bool(re.search(r"\b(?:fig(?:ure)?\.?\s*\d+|chart|graph|diagram|table)\b", f["text"], re.I))
                ),
            )
            if not self._is_low_value_summary_fact(fact)
        ]
        conclusion_candidates = [
            fact for fact in self._rank_facts(
                facts,
                predicate=lambda f: (
                    self._contains_any(f["section"], self._CONCLUSION_TERMS)
                    or self._contains_any(f["text"], self._CONCLUSION_TERMS)
                    or self._contains_any(f["text"], self._OUTLOOK_TERMS)
                ),
            )
            if not self._is_low_value_summary_fact(fact)
        ]

        groups: dict[str, list[dict[str, Any]]] = {
            "Purpose": [],
            "Major Findings": [],
            "Important Facts / Metrics": [],
            "Supporting Evidence": [],
            "Conclusion": [],
        }
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(label: str, fact: dict[str, Any] | None) -> None:
            if fact is None or fact["key"] in seen:
                return
            groups[label].append(fact)
            selected.append(fact)
            seen.add(fact["key"])

        add("Purpose", purpose_fact)

        # Cover as many distinct major sections as possible without overloading
        # any one section. Six to eight representatives works well for both
        # teaching PDFs and business reports.
        for fact in major[:8]:
            add("Major Findings", fact)

        for fact in metric_candidates:
            if len(groups["Important Facts / Metrics"]) >= 3:
                break
            add("Important Facts / Metrics", fact)

        supporting_pool = evidence_candidates + [
            fact for fact in representatives
            if fact["key"] not in seen
        ]
        for fact in supporting_pool:
            if len(groups["Supporting Evidence"]) >= 2:
                break
            add("Supporting Evidence", fact)

        conclusion_added = False
        for fact in conclusion_candidates:
            if fact["key"] not in seen:
                add("Conclusion", fact)
                conclusion_added = True
                break
        if not conclusion_added:
            tail = [
                f for f in sorted(facts, key=lambda x: x["position"], reverse=True)
                if not self._is_low_value_summary_fact(f)
                and not self._is_conversion_example_fact(f)
                and f["key"] not in seen
            ]
            add("Conclusion", tail[0] if tail else None)

        # Fill to the strict lower bound using unused high-level/source-order
        # facts. Conversion exercises, watermarks, examples and footer text are
        # excluded by the fact filters.
        fill_pool = major + [
            fact for fact in sorted(facts, key=lambda f: f["position"])
            if not self._is_low_value_summary_fact(fact)
            and not self._is_conversion_example_fact(fact)
        ]
        for fact in fill_pool:
            if self._summary_word_count(groups) >= self.FILE_SUMMARY_TARGET_WORDS:
                break
            add("Major Findings", fact)

        # Reduce only expandable middle groups until <=250 words.
        while self._summary_word_count(groups) > self.FILE_SUMMARY_MAX_WORDS:
            removed = False
            for label in ("Major Findings", "Important Facts / Metrics", "Supporting Evidence"):
                if len(groups[label]) > 1:
                    groups[label].pop()
                    removed = True
                    break
            if not removed:
                break

        # If removal dropped below 200, add the shortest unused clean facts.
        if self._summary_word_count(groups) < self.FILE_SUMMARY_MIN_WORDS:
            candidates = [
                fact for fact in fill_pool
                if fact["key"] not in seen
            ]
            candidates.sort(key=lambda f: (self._word_count(f["text"]), f["position"]))
            for fact in candidates:
                projected = self._summary_word_count(groups) + self._word_count(fact["text"])
                if projected > self.FILE_SUMMARY_MAX_WORDS:
                    continue
                add("Major Findings", fact)
                if self._summary_word_count(groups) >= self.FILE_SUMMARY_MIN_WORDS:
                    break

        text = self._render_summary_groups(groups)
        # Final safety: if a pathological source sentence makes the result too
        # long, trim at a sentence boundary. Never pad with invented content.
        if self._word_count(text) > self.FILE_SUMMARY_MAX_WORDS:
            text = self._trim_summary_to_limit(text, self.FILE_SUMMARY_MAX_WORDS)

        used_ids = self._supporting_ids_for_text(text, selected)
        return text, used_ids

    @classmethod
    def _pick_summary_purpose(
        cls,
        facts: Sequence[dict[str, Any]],
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_score = float("-inf")
        for fact in facts:
            if cls._is_low_value_summary_fact(fact) or cls._is_conversion_example_fact(fact):
                continue
            text = str(fact.get("text") or "")
            lower = text.lower()
            section = cls._compact(str(fact.get("section") or ""))
            score = 0.0
            if any(term in section for term in ("overview", "introduction", "company overview", "purpose", "objectives")):
                score += 12.0
            if any(
                phrase in lower
                for phrase in (
                    " is a process", " is a ", " are physical devices",
                    "focused on", "focuses on", "describes", "explains", "covers",
                    "introduce", "purpose",
                )
            ):
                score += 6.0
            if cls._contains_any(text, cls._PURPOSE_TERMS):
                score += 4.0
            if any(term in lower for term in ("disadvantage", "drawback", "failure", "example:", "e.g.")):
                score -= 10.0
            score -= min(int(fact.get("position", 0)), 100) * 0.03
            if score > best_score:
                best_score = score
                best = fact
        return best

    @classmethod
    def _is_conversion_example_fact(cls, fact: dict[str, Any]) -> bool:
        section = cls._compact(str(fact.get("section") or ""))
        text = str(fact.get("text") or "").lower()
        markers = (
            "binary to decimal", "decimal to binary", "octal to binary", "hexadecimal to binary",
            "positional weight", "we write 3 bit", "we write 4 bit", "subsequent quotient",
        )
        return any(marker in section or marker in text for marker in markers)

    @classmethod
    def _render_summary_groups(
        cls,
        groups: dict[str, list[dict[str, Any]]],
    ) -> str:
        parts: list[str] = []
        for label in (
            "Purpose",
            "Major Findings",
            "Important Facts / Metrics",
            "Supporting Evidence",
            "Conclusion",
        ):
            facts = groups.get(label, [])
            if not facts:
                continue
            body = " ".join(f["text"] for f in sorted(facts, key=lambda x: x["position"]))
            body = re.sub(r"\s+", " ", body).strip()
            if body:
                parts.append(f"{label}: {body}")
        return "\n\n".join(parts).strip()

    # =========================================================
    # EXECUTIVE SUMMARY: EXACTLY FIVE LINES
    # =========================================================

    def _executive_summary(
        self,
        evidence: Sequence[Dict[str, Any]],
    ) -> tuple[str, list[str]]:
        facts = self._facts(evidence)
        if not facts:
            line = "Insufficient verified evidence was available in the selected document."
            return "\n".join([
                f"Main Result: {line}",
                f"Key Metric / Finding: {line}",
                f"Major Insight: {line}",
                f"Risk / Opportunity: {line}",
                f"Final Takeaway: {line}",
            ]), []

        clean_facts = [
            fact for fact in facts
            if not self._is_low_value_summary_fact(fact)
            and not self._is_conversion_example_fact(fact)
        ] or facts

        purpose = self._pick_summary_purpose(clean_facts)
        representatives = self._high_level_section_representatives(clean_facts)
        if not representatives:
            representatives = self._section_representatives(clean_facts)

        metrics = self._rank_facts(clean_facts, predicate=lambda f: self._is_metric_fact(f))
        risks = self._rank_facts(
            clean_facts,
            predicate=lambda f: self._contains_any(f["text"], self._RISK_TERMS)
            or self._contains_any(f["section"], ("risk", "risks")),
        )
        opportunities = self._rank_facts(
            clean_facts,
            predicate=lambda f: self._contains_any(f["text"], self._OPPORTUNITY_TERMS)
            or self._contains_any(f["section"], ("outlook", "applications", "priorities")),
        )
        finals = self._rank_facts(
            clean_facts,
            predicate=lambda f: self._contains_any(f["section"], self._CONCLUSION_TERMS)
            or self._contains_any(f["text"], self._CONCLUSION_TERMS + self._OUTLOOK_TERMS),
        )

        used: set[str] = set()
        chosen: list[dict[str, Any]] = []

        def choose(candidates: Sequence[dict[str, Any]], fallback: Sequence[dict[str, Any]]) -> dict[str, Any]:
            for fact in list(candidates) + list(fallback):
                if fact["key"] not in used:
                    used.add(fact["key"])
                    chosen.append(fact)
                    return fact
            fact = clean_facts[0]
            chosen.append(fact)
            return fact

        main = purpose or choose(representatives, clean_facts)
        if main["key"] not in used:
            used.add(main["key"])
            chosen.append(main)

        metric = choose(metrics, representatives)
        insight_candidates = [
            fact for fact in representatives
            if fact["key"] not in used
            and not self._contains_any(
                fact["section"],
                ("overview", "introduction", "objectives", "company overview"),
            )
        ]
        def insight_priority(fact: dict[str, Any]) -> tuple[int, float, int]:
            section = self._compact(str(fact.get("section") or ""))
            preferred = (
                "network devices", "data flow", "topology", "protocols",
                "financial", "product", "customer", "operations", "regional",
            )
            history_penalty = 1 if "history" in section else 0
            section_bonus = 0 if any(term in section for term in preferred) else 1
            return (section_bonus + history_penalty, -fact["score"], fact["position"])

        insight_candidates.sort(key=insight_priority)
        insight = choose(insight_candidates, clean_facts)

        explicit_risks = [
            fact for fact in risks
            if self._contains_any(fact["section"], ("risk", "risks", "key risks"))
        ]
        strong_opportunities = [
            fact for fact in clean_facts
            if (
                self._contains_any(
                    fact["text"],
                    (
                        "benefit", "advantage", "applications", "offers multiple benefits",
                        "used for", "enables", "helps", "improve", "growth",
                        "target", "plan", "priority", "expansion",
                    ),
                )
                or self._contains_any(fact["section"], ("applications", "outlook", "priorities"))
            )
            and not self._contains_any(
                fact["text"],
                ("disadvantage", "drawback", "failure", "weakness", "issue", "problem", "limitation"),
            )
        ]
        strong_opportunities.sort(key=lambda f: (-f["score"], f["position"]))
        # Explicit risk sections win for business reports. Otherwise prefer an
        # actual supported benefit/application/opportunity instead of an
        # incidental topology sentence containing "failure" or "weakness".
        risk_or_opportunity = choose(
            explicit_risks or strong_opportunities or opportunities or risks,
            representatives,
        )
        final = choose(finals, list(reversed(representatives)) or list(reversed(clean_facts)))

        lines = [
            f"Main Result: {self._clip_words(main['text'], 28)}",
            f"Key Metric / Finding: {self._clip_words(metric['text'], 28)}",
            f"Major Insight: {self._clip_words(insight['text'], 28)}",
            f"Risk / Opportunity: {self._clip_words(risk_or_opportunity['text'], 28)}",
            f"Final Takeaway: {self._clip_words(final['text'], 28)}",
        ]
        used_ids = list(dict.fromkeys(
            str(fact.get("chunk_id") or "") for fact in chosen if str(fact.get("chunk_id") or "")
        ))
        return "\n".join(lines), used_ids

    # =========================================================
    # FACT EXTRACTION / CLEANING
    # =========================================================

    def _facts(self, evidence: Sequence[Dict[str, Any]]) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        seen: set[str] = set()
        position = 0
        current_section = "Document Highlights"

        low_value_sections = {
            "pre-unit preparatory material",
            "post-unit reading material",
            "glossary",
            "references",
            "bibliography",
            "table of contents",
        }

        def add_block(
            block_lines: list[str],
            *,
            section: str,
            chunk_id: str,
            metadata: Dict[str, Any],
            chunk_type: str,
        ) -> None:
            nonlocal position
            if not block_lines:
                return
            if self._compact(section) in low_value_sections:
                return
            block = " ".join(block_lines).strip()
            if not block:
                return
            for sentence in self._split_sentences(block):
                sentence = self._normalize_sentence(sentence)
                if not sentence:
                    continue
                word_count = self._word_count(sentence)
                if word_count < 5 or word_count > 70:
                    continue
                if self._looks_like_noise(sentence):
                    continue
                key = self._compact(sentence)
                if not key or key in seen:
                    continue
                seen.add(key)
                score = self._fact_score(
                    sentence,
                    section=section,
                    source_kind=chunk_type,
                    position=position,
                )
                facts.append(
                    {
                        "text": sentence,
                        "section": section,
                        "position": position,
                        "score": score,
                        "key": key,
                        "chunk_id": chunk_id,
                        "metadata": metadata,
                        "source_kind": chunk_type,
                    }
                )
                position += 1

        for item in sorted(evidence, key=self._source_order_key):
            metadata = dict(item.get("metadata") or {})
            chunk_type = str(metadata.get("chunk_type", "text") or "text").lower()
            content = self._clean_content(self._content(item))
            if not content:
                continue
            chunk_id = str(item.get("chunk_id") or "").strip()

            explicit_section = str(metadata.get("section", "") or "").strip()
            if explicit_section:
                current_section = explicit_section

            buffer: list[str] = []
            for raw_line in content.splitlines():
                line = re.sub(r"\s+", " ", raw_line).strip(" \t")
                if not line:
                    continue
                # Some PDF extractors flatten a heading and the first sentence
                # onto the same line (for example "Conclusion A data ...").
                inline_heading = re.match(
                    r"^(Conclusion|Overview|Introduction|Objectives|Learning Outcomes)\s+(.+)$",
                    line,
                    flags=re.I,
                )
                if inline_heading:
                    add_block(
                        buffer,
                        section=current_section,
                        chunk_id=chunk_id,
                        metadata=metadata,
                        chunk_type=chunk_type,
                    )
                    buffer = [inline_heading.group(2).strip()]
                    current_section = inline_heading.group(1).strip()
                    continue
                if self._is_section_heading_line(line):
                    add_block(
                        buffer,
                        section=current_section,
                        chunk_id=chunk_id,
                        metadata=metadata,
                        chunk_type=chunk_type,
                    )
                    buffer = []
                    current_section = self._normalize_heading(line)
                    continue
                buffer.append(line)

            add_block(
                buffer,
                section=current_section,
                chunk_id=chunk_id,
                metadata=metadata,
                chunk_type=chunk_type,
            )

        return facts

    @classmethod
    def _is_section_heading_line(cls, value: str) -> bool:
        text = str(value or "").strip()
        if not text or len(text) > 95:
            return False
        if re.match(r"^[•*-]", text):
            return False
        if re.match(r"^\d{1,2}\s*[\)]", text):
            return False
        if re.match(r"^Fig(?:ure)?\.?\s*\d+", text, flags=re.I):
            return False
        if re.search(r"[.!?]$", text):
            return False
        words = [w for w in text.split() if any(c.isalpha() for c in w)]
        if not (1 <= len(words) <= 11):
            return False
        # Numbered section titles such as "3. Product and Customer Performance".
        if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Za-z]", text):
            return True
        # A./B. subsection titles such as "A. Protocols".
        if re.match(r"^[A-Z]\.\s+[A-Za-z]", text):
            return True
        # Common report/course heading shape, including a single colon.
        stripped = text.replace(":", " ")
        title_words = [w for w in stripped.split() if any(c.isalpha() for c in w)]
        title_like = sum(1 for w in title_words if w[:1].isupper() or w.isupper())
        return title_like >= max(1, (len(title_words) + 1) // 2)

    @staticmethod
    def _normalize_heading(value: str) -> str:
        text = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", str(value or "")).strip()
        text = re.sub(r"^\s*[A-Z]\.\s+", "", text).strip()
        return text or "Document Highlights"

    @classmethod
    def _clean_content(cls, value: str) -> str:
        text = str(value or "").replace("\r", "\n")
        raw_lower = text.lower()
        # Entire TOC chunks are navigation metadata, not summary evidence.
        if "table of contents" in raw_lower:
            return ""
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        short_heading_lines = sum(
            1 for line in raw_lines
            if len(line.split()) <= 8 and not re.search(r"[.!?]$", line)
        )
        if (
            len(raw_lines) >= 6
            and short_heading_lines / max(1, len(raw_lines)) >= 0.72
            and any(term in raw_lower for term in ("glossary", "post-unit reading material", "objectives", "learning outcomes"))
        ):
            return ""
        clean_lines: list[str] = []
        for raw in text.splitlines():
            line = re.sub(r"\s+", " ", raw).strip(" \t•>-_=@π⑦⑧⑤°")
            if not line:
                continue
            lower = line.lower()
            if any(re.search(pattern, lower, flags=re.I) for pattern in cls._NOISE_LINE_PATTERNS):
                continue
            # Remove page headers that only repeat a course/report title.
            if re.fullmatch(r"page\s+\d+(?:\s*/\s*\d+)?", lower):
                continue
            clean_lines.append(line)

        text = "\n".join(clean_lines)
        # Some PDF chunks flatten headers/footers into a normal prose line.
        # Remove only known repeated footer fragments; substantive source text
        # is preserved verbatim.
        cleaned_lines: list[str] = []
        for line in text.splitlines():
            line = re.sub(r"Powered by Great Learning\.?", " ", line, flags=re.I)
            line = re.sub(r"Proprietary content\.?[^$]*$", " ", line, flags=re.I)
            line = re.sub(r"This file is meant for personal use.*$", " ", line, flags=re.I)
            line = re.sub(r"Sharing or publishing the contents.*$", " ", line, flags=re.I)
            line = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " ", line)
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)
        text = cls._strip_source_markers(text)
        return text.strip()

    @classmethod
    def _section_name(cls, item: Dict[str, Any], content: str) -> str:
        metadata = dict(item.get("metadata") or {})
        section = str(metadata.get("section", "") or "").strip()
        if section:
            return section

        first = content.splitlines()[0].strip() if content else ""
        if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Za-z]", first):
            return first
        if 1 <= len(first.split()) <= 8 and cls._looks_like_heading(first):
            return first
        return "Document Highlights"

    @staticmethod
    def _looks_like_heading(value: str) -> bool:
        text = str(value or "").strip()
        if not text or len(text) > 80 or text[-1:] in ".!?;":
            return False
        words = [w for w in text.split() if any(c.isalpha() for c in w)]
        if not words:
            return False
        title_like = sum(1 for w in words if w[:1].isupper() or w.isupper())
        return title_like >= max(1, (len(words) + 1) // 2)

    @classmethod
    def _split_sentences(cls, value: str) -> list[str]:
        text = str(value or "").replace("\r", "\n")
        # Keep numbered/bullet items as separate facts.
        text = re.sub(r"\n\s*[•*-]\s*", ". ", text)
        text = re.sub(r"\n\s*(\d{1,2}\s*[\).])\s*", r". \1 ", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        protected = text
        replacements = {
            "Ltd.": "Ltd<prd>",
            "Inc.": "Inc<prd>",
            "Co.": "Co<prd>",
            "Dr.": "Dr<prd>",
            "Mr.": "Mr<prd>",
            "Ms.": "Ms<prd>",
            "e.g.": "e<prd>g<prd>",
            "i.e.": "i<prd>e<prd>",
        }
        for source, target in replacements.items():
            protected = protected.replace(source, target)

        parts = [
            part.strip().replace("<prd>", ".")
            for part in re.split(
                r"(?<=[.!?])\s+|(?=\b\d{1,2}\s*[\).]\s+[A-Z])",
                protected,
            )
            if part.strip()
        ]
        return parts

    @classmethod
    def _fact_score(
        cls,
        text: str,
        *,
        section: str,
        source_kind: str,
        position: int,
    ) -> float:
        lowered = text.lower()
        score = 1.0
        if re.search(r"\d", text):
            score += 1.2
        if cls._contains_any(text, cls._SALIENT_TERMS):
            score += 1.5
        if cls._contains_any(text, cls._PURPOSE_TERMS):
            score += 0.7
        if any(marker in lowered for marker in ("chart should be interpreted", "figure 1 summarizes", "figure summarizes")):
            score -= 2.0
        if cls._contains_any(text, cls._RISK_TERMS):
            score += 0.8
        if cls._contains_any(text, cls._OUTLOOK_TERMS):
            score += 0.7
        if source_kind in {"table", "chart", "image"}:
            score += 0.5
        if cls._contains_any(section, cls._CONCLUSION_TERMS):
            score += 0.8
        # Small source-order preference prevents late footer-like text from
        # outranking the substantive beginning of a document.
        score += max(0.0, 0.30 - min(position, 30) * 0.01)
        return score

    @classmethod
    def _section_representatives(
        cls,
        facts: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        section_order: list[str] = []
        for fact in facts:
            section = str(fact.get("section") or "Document Highlights")
            compact = cls._compact(section)
            if (
                compact.startswith("ex ")
                or compact in {"text", "numbers", "images", "audio", "video", "glossary", "pre unit preparatory material", "post unit reading material", "learning outcomes", "objectives"}
            ):
                continue
            if section not in grouped:
                grouped[section] = []
                section_order.append(section)
            grouped[section].append(fact)

        output: list[dict[str, Any]] = []
        for section in section_order:
            values = grouped[section]
            if not values:
                continue
            # Prefer a clear defining/overview sentence near the beginning of
            # the section instead of a later drawback/example with more digits.
            defining = [
                fact for fact in values[:6]
                if re.search(r"\b(?:is|are|means|consists|used|refers|contains|has|provides)\b", fact["text"], flags=re.I)
            ]
            chosen = defining[0] if defining else max(
                values[:6],
                key=lambda f: (f["score"], -f["position"]),
            )
            output.append(chosen)
        return output

    @classmethod
    def _is_low_value_summary_fact(cls, fact: dict[str, Any]) -> bool:
        section = cls._compact(str(fact.get("section") or ""))
        text = str(fact.get("text") or "").lower()
        if any(cls._compact(term) in section for term in cls._LOW_LEVEL_SECTION_TERMS):
            return True
        if re.match(r"^ex(?:ample)?\.?\s*\d*", section):
            return True
        if any(marker in text for marker in ("we write 3-bit", "we write 4-bit", "subsequent quotient", "positional weight value")):
            return True
        if re.search(r"(?:as follows|listed below)\s*:\s*\d*\.?$", text.strip(), flags=re.I):
            return True
        return False

    @classmethod
    def _is_high_level_section_name(cls, section: str) -> bool:
        compact = cls._compact(section)
        if not compact:
            return False

        exact = {
            "overview", "introduction", "company overview",
            "components of data communication", "data representation", "data flow",
            "network devices", "types of connection", "topology",
            "categories of networks", "interconnection of networks applications of networks",
            "history of network", "protocols", "standards", "protocols and standards",
            "conclusion", "overall assessment",
        }
        if compact in exact:
            return True

        # Report-style high-level sections often carry numbering or qualifiers.
        report_terms = (
            "financial", "sales performance", "product and customer",
            "operations and supply chain", "key risks", "regional performance",
            "outlook", "overall assessment",
        )
        return any(term in compact for term in report_terms)

    @classmethod
    def _high_level_section_representatives(
        cls,
        facts: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        section_order: list[str] = []
        for fact in facts:
            section = str(fact.get("section") or "Document Highlights")
            compact = cls._compact(section)
            if not compact or cls._is_low_value_summary_fact(fact):
                continue
            if not cls._is_high_level_section_name(section):
                continue
            if section not in grouped:
                grouped[section] = []
                section_order.append(section)
            grouped[section].append(fact)

        output: list[dict[str, Any]] = []
        for section in section_order:
            values = grouped[section]

            section_terms = {
                token
                for token in re.findall(r"[a-z0-9]+", section.lower())
                if len(token) > 3
                and token not in {"with", "from", "into", "about", "network", "performance"}
            }

            def representative_score(fact: dict[str, Any]) -> tuple[float, int]:
                text = str(fact.get("text") or "")
                lower = text.lower()
                score = float(fact.get("score", 0.0) or 0.0)
                text_terms = set(re.findall(r"[a-z0-9]+", lower))
                score += len(section_terms & text_terms) * 2.5
                if re.search(
                    r"\b(?:is|are|means|consists|contains|has|used|refers|provides|reached|generated|improved|remained|targeting)\b",
                    text,
                    flags=re.I,
                ):
                    score += 1.5
                if any(
                    marker in lower
                    for marker in (
                        "five elements", "types of", "three basic categories",
                        "physical devices", "process of exchanging", "data will flow",
                        "four types", "standards regulate",
                    )
                ):
                    score += 3.0
                if "components" in section.lower() and "elements" in lower:
                    score += 4.0
                if "history" in section.lower() and "began" in lower:
                    score += 4.0
                if "applications" in section.lower() and "benefit" in lower:
                    score += 4.0
                if re.match(r"^[o•*-]\s*", text):
                    score -= 2.0
                if re.match(r"^\d{1,2}\s*[\).]", text):
                    score -= 4.0
                if text.lower().startswith("example"):
                    score -= 2.0
                return score, -int(fact.get("position", 0) or 0)

            chosen = max(values[:8], key=representative_score)
            output.append(chosen)
        return output

    @classmethod
    def _spread_facts(
        cls,
        facts: Sequence[dict[str, Any]],
        *,
        segments: int = 6,
    ) -> list[dict[str, Any]]:
        """Choose informative facts across the full source order."""
        values = sorted(facts, key=lambda f: f["position"])
        if not values:
            return []
        segments = max(1, min(int(segments), len(values)))
        output: list[dict[str, Any]] = []
        for index in range(segments):
            start = round(index * len(values) / segments)
            end = round((index + 1) * len(values) / segments)
            bucket = values[start:end] or values[start:start + 1]
            if not bucket:
                continue
            chosen = max(bucket, key=lambda f: (f["score"], -f["position"]))
            output.append(chosen)
        return output

    @classmethod
    def _is_metric_fact(cls, fact: dict[str, Any]) -> bool:
        text = str(fact.get("text") or "")
        if not cls._contains_any(text, cls._METRIC_TERMS):
            return False
        numbers = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text)
        non_year_numbers = []
        for value in numbers:
            try:
                numeric = float(value.rstrip("%").replace(",", ""))
            except ValueError:
                continue
            if 1900 <= numeric <= 2100 and "%" not in value:
                continue
            non_year_numbers.append(value)
        structural_terms = ("elements", "types", "categories", "components", "modes")
        number_words = bool(
            re.search(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b", text, flags=re.I)
            and cls._contains_any(text, structural_terms)
        )
        return bool(non_year_numbers or number_words)

    @classmethod
    def _rank_facts(
        cls,
        facts: Sequence[dict[str, Any]],
        *,
        predicate,
    ) -> list[dict[str, Any]]:
        values = [fact for fact in facts if predicate(fact)]
        return sorted(values, key=lambda f: (-f["score"], f["position"]))

    @classmethod
    def _looks_like_noise(cls, sentence: str) -> bool:
        lower = sentence.lower()
        if any(re.search(pattern, lower, flags=re.I) for pattern in cls._NOISE_LINE_PATTERNS):
            return True
        # Cover/TOC metadata often arrives flattened into one long sentence.
        if sum(marker in lower for marker in ("program:", "specialization:", "semester:", "course name:", "course code:", "unit name:")) >= 3:
            return True
        if lower.count("topology") >= 5 and "table of contents" in lower:
            return True
        return False

    @staticmethod
    def _normalize_sentence(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip(" \t-•")
        if not text:
            return ""
        if text[-1] not in ".!?":
            text += "."
        return text

    # =========================================================
    # NON-UI / SECTION SCOPE: EXISTING MODEL-COMPATIBLE PATH
    # =========================================================

    def _existing_scope_summary(
        self,
        *,
        evidence: Sequence[Dict[str, Any]],
        scope: str,
        title: str,
    ) -> str:
        compact = self._compact_evidence(evidence)
        prompt = (
            "You are the summarization stage of DocMindAI. Use ONLY the "
            "supplied retrieved evidence. Do not invent facts. Preserve important "
            "numbers exactly. Do not write [Source: ...] markers because the UI "
            "renders citations separately.\n\n"
            f"Summary scope: {scope}\n"
            f"Document/section: {title or 'Selected evidence'}\n\n"
            f"Evidence:\n{self._format_evidence(compact)}"
        )

        if self.reasoning_agent is None:
            return self._safe_extract(compact, max_words=300)

        llm = getattr(self.reasoning_agent, "llm", None)
        if llm is not None:
            try:
                return self._strip_source_markers(
                    str(
                        llm.chat(
                            prompt=prompt,
                            images=[],
                            temperature=0.0,
                            num_predict=self.SECTION_NUM_PREDICT,
                            num_ctx=self.SECTION_NUM_CTX,
                        )
                    ).strip()
                )
            except Exception:
                return self._safe_extract(compact, max_words=300)

        try:
            result = self.reasoning_agent.reason(
                prompt,
                compact,
                validation={"valid_evidence": list(compact)},
            )
        except Exception:
            return self._safe_extract(compact, max_words=300)

        if isinstance(result, dict):
            return self._strip_source_markers(
                str(
                    result.get("answer")
                    or result.get("content")
                    or result.get("response")
                    or ""
                ).strip()
            )
        return self._strip_source_markers(str(result or "").strip())

    # =========================================================
    # CITATIONS / SUPPORT HELPERS
    # =========================================================

    @classmethod
    def _supporting_ids_for_text(
        cls,
        text: str,
        selected: Sequence[dict[str, Any]],
    ) -> list[str]:
        compact_text = cls._compact(text)
        ids: list[str] = []
        for fact in selected:
            key = fact.get("key", "")
            chunk_id = str(fact.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            # A fact is supporting if a distinctive prefix survives rendering.
            tokens = key.split()
            probe = " ".join(tokens[: min(8, len(tokens))])
            if probe and probe in compact_text:
                ids.append(chunk_id)
        return list(dict.fromkeys(ids))

    @staticmethod
    def _evidence_by_ids(
        evidence: Sequence[Dict[str, Any]],
        ids: Sequence[str],
    ) -> list[Dict[str, Any]]:
        wanted = {str(value) for value in ids if str(value)}
        return [
            item for item in evidence
            if str(item.get("chunk_id") or "") in wanted
        ]

    @staticmethod
    def _citations(evidence: Sequence[Dict[str, Any]]) -> List[str]:
        citations: list[str] = []
        seen: set[str] = set()
        for item in evidence:
            metadata = dict(item.get("metadata") or {})
            filename = metadata.get("filename") or metadata.get("file_name")
            if not filename:
                continue
            page = metadata.get("page_number")
            sheet = metadata.get("sheet_name")
            table_id = metadata.get("table_id")
            if page is not None:
                value = f"[Source: {filename}, Page {page}]"
            elif sheet:
                value = f"[Source: {filename}, Sheet: {sheet}]"
            elif table_id:
                value = f"[Source: {filename}, Table {table_id}]"
            else:
                value = f"[Source: {filename}]"
            if value not in seen:
                seen.add(value)
                citations.append(value)
        return citations

    # =========================================================
    # GENERIC HELPERS
    # =========================================================

    @staticmethod
    def _content(item: Dict[str, Any]) -> str:
        return str(item.get("content", item.get("text", "")) or "").strip()

    @staticmethod
    def _compact(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    @staticmethod
    def _word_count(value: str) -> int:
        return len(re.findall(r"\b[\w'-]+\b", str(value or "")))

    @classmethod
    def _summary_word_count(cls, groups: dict[str, list[dict[str, Any]]]) -> int:
        value = cls._render_summary_groups(groups)
        return cls._word_count(value)

    @staticmethod
    def _contains_any(value: str, terms: Sequence[str]) -> bool:
        lower = str(value or "").lower()
        return any(term in lower for term in terms)

    @classmethod
    def _trim_summary_to_limit(cls, value: str, max_words: int) -> str:
        words = str(value or "").split()
        if len(words) <= max_words:
            return str(value or "").strip()
        trimmed = " ".join(words[:max_words]).strip()
        end = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
        if end >= int(len(trimmed) * 0.78):
            trimmed = trimmed[: end + 1].strip()
        return trimmed

    @classmethod
    def _clip_words(cls, value: str, max_words: int) -> str:
        words = str(value or "").split()
        if len(words) <= max_words:
            return str(value or "").strip()
        return " ".join(words[:max_words]).rstrip(" ,;:") + "…"

    @classmethod
    def _safe_extract(
        cls,
        evidence: Sequence[Dict[str, Any]],
        *,
        max_words: int,
    ) -> str:
        output: list[str] = []
        seen: set[str] = set()
        count = 0
        for item in sorted(evidence, key=cls._source_order_key):
            content = cls._clean_content(cls._content(item))
            for sentence in cls._split_sentences(content):
                sentence = cls._normalize_sentence(sentence)
                key = cls._compact(sentence)
                if not key or key in seen or cls._looks_like_noise(sentence):
                    continue
                words = cls._word_count(sentence)
                if words < 5:
                    continue
                if count + words > max_words:
                    return " ".join(output).strip()
                seen.add(key)
                output.append(sentence)
                count += words
        return " ".join(output).strip()

    @classmethod
    def _compact_evidence(
        cls,
        evidence: Sequence[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        output: list[Dict[str, Any]] = []
        total = 0
        for item in evidence[:10]:
            content = cls._clean_content(cls._content(item))
            if not content:
                continue
            remaining = 7000 - total
            if remaining <= 0:
                break
            content = content[: min(850, remaining)]
            normalized = dict(item)
            normalized["content"] = content
            output.append(normalized)
            total += len(content)
        return output

    @classmethod
    def _format_evidence(cls, evidence: Sequence[Dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, item in enumerate(evidence, start=1):
            metadata = dict(item.get("metadata") or {})
            blocks.append(
                "\n".join(
                    [
                        f"EVIDENCE {index}",
                        "DOCUMENT: " + str(
                            metadata.get("filename")
                            or metadata.get("file_name")
                            or "Unknown"
                        ),
                        "PAGE: " + str(metadata.get("page_number", "")),
                        "CONTENT: " + cls._clean_content(cls._content(item)),
                    ]
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _source_order_key(item: Dict[str, Any]) -> tuple:
        metadata = dict(item.get("metadata") or {})

        def as_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        return (
            str(metadata.get("document_id", "")),
            as_int(metadata.get("chunk_index"), 10**9),
            as_int(metadata.get("page_number"), 10**9),
            str(item.get("chunk_id", "")),
        )

    @staticmethod
    def _strip_source_markers(value: str) -> str:
        text = str(value or "")
        text = re.sub(r"\[\s*Source\s*:[^\]]+\]", "", text, flags=re.I)
        text = re.sub(
            r"(?is)\n?\s*Sources\s*/?\s*Citations\s*:?.*$",
            "",
            text,
        )
        return re.sub(r"[ \t]+\n", "\n", text).strip()

    @staticmethod
    def _all_ids(evidence: Sequence[Dict[str, Any]]) -> list[str]:
        return list(
            dict.fromkeys(
                str(item.get("chunk_id"))
                for item in evidence
                if item.get("chunk_id") is not None
            )
        )

    @staticmethod
    def _document_count(evidence: Sequence[Dict[str, Any]]) -> int:
        documents: set[str] = set()
        for item in evidence:
            metadata = dict(item.get("metadata") or {})
            identity = (
                metadata.get("document_id")
                or metadata.get("filename")
                or metadata.get("file_name")
            )
            if identity:
                documents.add(str(identity))
        return len(documents)
