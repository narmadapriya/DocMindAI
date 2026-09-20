from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.rag.embeddings import EmbeddingModel
from app.rag.vectordb.search import VectorSearch
from app.core.performance import RAG_MAX_TOP_K


class Retriever:
    """
    Phase 9 multimodal retrieval service.

    The frozen retrieval flow is preserved. Narrow coverage guards are
    applied only to the source types that still need correction:

      * CSV/XLSX analytical questions: keep one complete table chunk in top-k.
      * PDF visual/page questions: keep the requested page visual chunk in top-k.
      * PDF detail/list questions: add only the local heading/page neighborhood needed for complete lists.
      * TXT questions: use the already-retrieved candidates and prefer chunks
        that cover the query terms/numbers, without adding another vector call.

    DOCX and working CSV behavior remain on their frozen paths.
    """

    _STRUCTURED_TERMS = (
        "highest",
        "lowest",
        "maximum",
        "minimum",
        "largest",
        "smallest",
        "increase",
        "increased",
        "decrease",
        "decreased",
        "difference",
        "change",
        "percentage",
        "percent",
        "average",
        "mean",
        "total",
        "sum",
        "combined",
        "between",
        "previous month",
        "previous row",
        "preceding month",
        "prior month",
        "more than",
        "less than",
        "current stock",
        "in production",
        "units sold",
    )

    _VISUAL_TERMS = (
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

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_search: Optional[VectorSearch] = None,
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.search = vector_search or VectorSearch()

    # =========================================================
    # PUBLIC RETRIEVE
    # =========================================================

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        document_ids: Optional[Sequence[str]] = None,
        modalities: Optional[Sequence[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        min_relevance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        if not user_id:
            raise ValueError("user_id is required for retrieval.")

        cleaned_query = query.strip()
        top_k = min(RAG_MAX_TOP_K, max(1, int(top_k)))

        # -----------------------------------------------------
        # 1. Embed rewritten query
        # -----------------------------------------------------
        query_embedding = self.embedding_model.embed(cleaned_query)

        # -----------------------------------------------------
        # 2. Build secure Chroma filters
        # -----------------------------------------------------
        filters = dict(metadata_filters or {})
        filters["user_id"] = user_id

        if document_ids:
            filters["document_id"] = {"$in": list(document_ids)}

        chroma_where = self._build_chroma_where(filters)

        # -----------------------------------------------------
        # 3. Frozen primary semantic retrieval
        # -----------------------------------------------------
        search_k = min(RAG_MAX_TOP_K, max(top_k, 10))

        primary_results = self.search.search(
            query_embedding=query_embedding,
            top_k=search_k,
            metadata_filters=chroma_where,
        )

        source_kinds = self._source_kinds(primary_results)
        structured_target = bool(
            source_kinds & {"csv", "xlsx"}
            and self._is_structured_query(cleaned_query)
        )
        pdf_visual_target = bool(
            "pdf" in source_kinds
            and self._is_visual_query(cleaned_query)
        )
        pdf_text_precision_target = bool(
            "pdf" in source_kinds
            and not pdf_visual_target
            and self._is_pdf_text_precision_query(cleaned_query)
        )
        txt_target = "txt" in source_kinds

        # -----------------------------------------------------
        # DOCX and every unaffected source keep the original
        # frozen ranking/selection behavior exactly.
        # -----------------------------------------------------
        if (
            not structured_target
            and not pdf_visual_target
            and not pdf_text_precision_target
            and not txt_target
        ):
            return self._baseline_select(
                primary_results,
                modalities=modalities,
                top_k=top_k,
                min_relevance=min_relevance,
            )

        # -----------------------------------------------------
        # 4. Narrow supplemental coverage only for CSV/XLSX/PDF
        # -----------------------------------------------------
        supplemental: list[Dict[str, Any]] = []

        if structured_target:
            supplemental.extend(
                self._retrieve_chunk_type(
                    query_embedding=query_embedding,
                    filters=filters,
                    chunk_type="table",
                    top_k=min(4, RAG_MAX_TOP_K),
                )
            )

        explicit_page = self._explicit_page_number(cleaned_query)
        pdf_text_focus_page: int | None = None
        pdf_text_focus_pages: list[int] = []

        if pdf_text_precision_target:
            # PDF-text-only precision coverage. This is deliberately isolated
            # from DOCX/CSV/XLSX/TXT and from the already-working PDF visual
            # path. Extremum questions need the complete PDF table; explicit
            # factor/driver questions need all text chunks from the page that
            # contains the matching list heading.
            if self._is_pdf_table_extremum_query(cleaned_query):
                supplemental.extend(
                    self._retrieve_chunk_type(
                        query_embedding=query_embedding,
                        filters=filters,
                        chunk_type="table",
                        top_k=min(4, RAG_MAX_TOP_K),
                    )
                )

            if self._is_pdf_list_query(cleaned_query):
                pdf_text_focus_page = self._pdf_list_focus_page(
                    primary_results,
                    cleaned_query,
                )
                if pdf_text_focus_page is not None:
                    # List/detail content often starts at the end of one page
                    # and continues onto the next page.  Include the immediate
                    # three-page neighborhood only for this narrow PDF query
                    # shape; all other retrieval behaviour stays frozen.
                    pdf_text_focus_pages = [
                        page
                        for page in (
                            pdf_text_focus_page - 1,
                            pdf_text_focus_page,
                            pdf_text_focus_page + 1,
                        )
                        if page >= 1
                    ]
                    # Exact page coverage is important for explicit lists.
                    # Semantic search can return only one fragment from a page,
                    # which is not enough when a numbered list spans several
                    # chunks. Fetch all authenticated text chunks from the local
                    # page neighborhood first, then retain semantic search as a
                    # fallback. This branch is PDF-list-only.
                    supplemental.extend(
                        self._retrieve_pdf_page_text_chunks(
                            filters=filters,
                            page_numbers=pdf_text_focus_pages,
                        )
                    )
                    for page_number in pdf_text_focus_pages:
                        page_filters = dict(filters)
                        page_filters["page_number"] = page_number
                        page_filters["chunk_type"] = "text"
                        supplemental.extend(
                            self.search.search(
                                query_embedding=query_embedding,
                                top_k=min(6, RAG_MAX_TOP_K),
                                metadata_filters=self._build_chroma_where(page_filters),
                            )
                        )

        if pdf_visual_target:
            # Pull real PDF visual chunks even if Router supplied only text.
            for chunk_type in ("image", "chart"):
                visual_filters = dict(filters)
                visual_filters["chunk_type"] = chunk_type
                if explicit_page is not None:
                    visual_filters["page_number"] = explicit_page

                supplemental.extend(
                    self.search.search(
                        query_embedding=query_embedding,
                        top_k=min(3, RAG_MAX_TOP_K),
                        metadata_filters=self._build_chroma_where(
                            visual_filters
                        ),
                    )
                )

            if explicit_page is not None:
                page_filters = dict(filters)
                page_filters["page_number"] = explicit_page
                supplemental.extend(
                    self.search.search(
                        query_embedding=query_embedding,
                        top_k=min(4, RAG_MAX_TOP_K),
                        metadata_filters=self._build_chroma_where(
                            page_filters
                        ),
                    )
                )

        # -----------------------------------------------------
        # 5. Merge/deduplicate by chunk_id
        # -----------------------------------------------------
        merged: Dict[str, Dict[str, Any]] = {}

        for result in list(primary_results) + supplemental:
            chunk_id = str(result.get("chunk_id", "")).strip()
            if not chunk_id:
                continue

            previous = merged.get(chunk_id)
            if previous is None or self._score(result) > self._score(previous):
                merged[chunk_id] = result

        # Ordinary candidates keep the frozen relevance threshold. For an
        # explicit PDF visual/page query, however, the metadata-scoped image
        # candidate is allowed to survive a low semantic score: image chunks
        # can have sparse OCR/vision text even though page_number + chunk_type
        # identifies the exact visual the user requested. The candidate is
        # still constrained by authenticated user/document filters above.
        pdf_visual_required_pool: list[Dict[str, Any]] = []
        if pdf_visual_target:
            pdf_visual_required_pool = [
                item
                for item in merged.values()
                if (
                    self._source_kind(item) == "pdf"
                    and self._chunk_type(item) in {"image", "chart"}
                    and (
                        explicit_page is None
                        or self._page_number(item) == explicit_page
                    )
                    and self._has_visual_asset_metadata(item)
                )
            ]

        results = [
            item
            for item in merged.values()
            if self._score(item) >= float(min_relevance)
        ]

        requested = {
            str(value).lower()
            for value in (modalities or [])
            if str(value).strip()
        }

        results.sort(
            key=lambda item: (
                self._score(item),
                self._requested_modality_match(item, requested),
            ),
            reverse=True,
        )

        selected = (
            self._select_txt_coverage(
                results=results,
                query=cleaned_query,
                top_k=top_k,
            )
            if txt_target
            else list(results[:top_k])
        )

        # -----------------------------------------------------
        # 6. Guarantee one complete CSV/XLSX table when needed.
        # -----------------------------------------------------
        required: list[Dict[str, Any]] = []

        if structured_target:
            table = self._best_candidate(
                results,
                lambda item: (
                    self._chunk_type(item) == "table"
                    and self._source_kind(item) in {"csv", "xlsx"}
                ),
            )
            if table is not None:
                required.append(table)

        # -----------------------------------------------------
        # 7. Guarantee PDF text evidence only for the narrow PDF
        #    precision questions that are still under correction.
        # -----------------------------------------------------
        if pdf_text_precision_target:
            if self._is_pdf_table_extremum_query(cleaned_query):
                table = self._best_candidate(
                    results,
                    lambda item: (
                        self._source_kind(item) == "pdf"
                        and self._chunk_type(item) == "table"
                    ),
                )
                if table is not None:
                    required.append(table)

            if (
                self._is_pdf_list_query(cleaned_query)
                and pdf_text_focus_page is not None
            ):
                neighborhood = [
                    item
                    for item in results
                    if (
                        self._source_kind(item) == "pdf"
                        and self._chunk_type(item) == "text"
                        and self._page_number(item) in set(pdf_text_focus_pages or [pdf_text_focus_page])
                    )
                ]
                neighborhood.sort(
                    key=lambda item: (
                        self._page_number(item) or 10**9,
                        int((item.get("metadata") or {}).get("chunk_index") or 0),
                    )
                )

                # Prefer chunks that contain the requested subject and explicit
                # numbered/bullet list members, then preserve source order.
                subject_terms = self._pdf_query_subject_terms(cleaned_query)
                ranked_neighborhood = sorted(
                    neighborhood,
                    key=lambda item: (
                        -self._pdf_list_chunk_priority(item, subject_terms),
                        self._page_number(item) or 10**9,
                        int((item.get("metadata") or {}).get("chunk_index") or 0),
                    ),
                )
                required.extend(ranked_neighborhood[:top_k])

        # -----------------------------------------------------
        # 8. Guarantee the requested PDF page visual when needed.
        # -----------------------------------------------------
        if pdf_visual_target:
            # Prefer a real local image asset even when its embedding score is
            # below min_relevance. Explicit page/chunk metadata is stronger
            # routing evidence for a visual question than sparse image text.
            # This is PDF-visual-only and cannot affect DOCX/CSV/XLSX/TXT.
            visual = self._best_candidate(
                pdf_visual_required_pool,
                lambda item: True,
            )

            if visual is None:
                visual = self._best_candidate(
                    results,
                    lambda item: (
                        self._source_kind(item) == "pdf"
                        and self._chunk_type(item) in {"image", "chart"}
                        and (
                            explicit_page is None
                            or self._page_number(item) == explicit_page
                        )
                        and self._has_visual_asset_metadata(item)
                    ),
                )

            if visual is None:
                visual = self._best_candidate(
                    results,
                    lambda item: (
                        self._source_kind(item) == "pdf"
                        and self._chunk_type(item) in {"image", "chart"}
                        and (
                            explicit_page is None
                            or self._page_number(item) == explicit_page
                        )
                    ),
                )

            if visual is not None:
                required.append(visual)

        # Explicit PDF list/detail questions sometimes require more than the
        # normal QA top-k because a single page can be split into several
        # chunks. Only this narrow branch may expand the evidence window.
        effective_top_k = top_k
        if (
            pdf_text_precision_target
            and self._is_pdf_list_query(cleaned_query)
            and pdf_text_focus_page is not None
        ):
            effective_top_k = max(top_k, min(12, max(8, len(required))))

        selected = self._ensure_required(
            selected=selected,
            required=required,
            top_k=effective_top_k,
        )

        selected.sort(
            key=lambda item: (
                self._score(item),
                self._requested_modality_match(item, requested),
            ),
            reverse=True,
        )

        return selected[:effective_top_k]

    # =========================================================
    # FROZEN BASELINE SELECTION
    # =========================================================

    @staticmethod
    def _baseline_select(
        results: Sequence[Dict[str, Any]],
        *,
        modalities: Optional[Sequence[str]],
        top_k: int,
        min_relevance: float,
    ) -> List[Dict[str, Any]]:
        filtered = [
            result
            for result in results
            if result.get("relevance_score", 0.0) >= min_relevance
        ]

        if modalities:
            requested = {
                str(modality).lower()
                for modality in modalities
            }
            filtered.sort(
                key=lambda item: (
                    item.get("relevance_score", 0.0),
                    1
                    if str(
                        item.get("metadata", {}).get("chunk_type", "")
                    ).lower()
                    in requested
                    else 0,
                ),
                reverse=True,
            )
        else:
            filtered.sort(
                key=lambda item: item.get("relevance_score", 0.0),
                reverse=True,
            )

        return filtered[:top_k]

    # =========================================================
    # COVERAGE HELPERS
    # =========================================================

    @staticmethod
    def _score(item: Dict[str, Any]) -> float:
        try:
            return float(item.get("relevance_score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _chunk_type(item: Dict[str, Any]) -> str:
        return str(
            (item.get("metadata") or {}).get("chunk_type", "")
        ).lower()

    @classmethod
    def _source_kind(cls, item: Dict[str, Any]) -> str:
        metadata = item.get("metadata") or {}

        filename = str(
            metadata.get("filename")
            or metadata.get("file_name")
            or metadata.get("original_filename")
            or ""
        ).strip()

        suffix = Path(filename).suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix in {".xlsx", ".xls"}:
            return "xlsx"
        if suffix == ".pdf":
            return "pdf"
        if suffix == ".docx":
            return "docx"
        if suffix == ".txt":
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
        results: Sequence[Dict[str, Any]],
    ) -> set[str]:
        return {
            kind
            for kind in (cls._source_kind(item) for item in results)
            if kind
        }

    @classmethod
    def _is_structured_query(cls, query: str) -> bool:
        q = re.sub(r"\s+", " ", str(query or "").lower()).strip()
        return any(term in q for term in cls._STRUCTURED_TERMS)

    @classmethod
    def _is_visual_query(cls, query: str) -> bool:
        q = str(query or "").lower()
        return any(term in q for term in cls._VISUAL_TERMS)

    @classmethod
    def _is_pdf_text_precision_query(cls, query: str) -> bool:
        q = re.sub(r"\s+", " ", str(query or "").lower()).strip()
        return cls._is_pdf_table_extremum_query(q) or cls._is_pdf_list_query(q)

    @staticmethod
    def _is_pdf_table_extremum_query(query: str) -> bool:
        q = re.sub(r"\s+", " ", str(query or "").lower()).strip()
        extremum = any(
            term in q
            for term in (
                "highest", "lowest", "maximum", "minimum", "largest", "smallest"
            )
        )
        metric = any(
            term in q
            for term in ("price", "revenue", "units", "sales", "share", "score")
        )
        return extremum and metric

    @staticmethod
    def _is_pdf_list_query(query: str) -> bool:
        q = re.sub(r"\s+", " ", str(query or "").lower()).strip()
        if any(term in q for term in ("factors", "drivers", "reasons", "causes", "priorities")):
            return True

        asks_for_collection = any(
            term in q
            for term in (
                "types", "type of", "kinds", "kind of", "categories",
                "category", "components", "features", "steps",
            )
        )
        asks_for_detail = any(
            term in q
            for term in (
                "explain", "describe", "in detail", "detail", "what are",
                "list", "give",
            )
        )
        return asks_for_collection and asks_for_detail

    @staticmethod
    def _pdf_query_subject_terms(query: str) -> set[str]:
        stop = {
            "the", "a", "an", "and", "or", "of", "to", "in", "on", "is",
            "are", "was", "were", "its", "it", "this", "that", "what",
            "which", "how", "explain", "describe", "give", "from", "with",
            "for", "about", "detail", "detailed", "types", "type", "kinds",
            "kind", "categories", "category", "components", "features",
            "steps", "list", "different", "common",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(query or "").lower())
            if len(token) > 2 and token not in stop
        }

    @classmethod
    def _pdf_list_chunk_priority(
        cls,
        item: Dict[str, Any],
        subject_terms: set[str],
    ) -> float:
        content = str(item.get("content", "") or "")
        lower = content.lower()
        score = 0.0
        score += sum(2.0 for term in subject_terms if term in lower)
        if re.search(r"(?<!\d)\d{1,2}\s*[\).]\s*[A-Za-z]", content):
            score += 4.0
        if any(marker in lower for marker in ("types of", "categories of", "components of")):
            score += 3.0
        if any(marker in lower for marker in ("network devices", "network device", "key factors", "key drivers")):
            score += 2.0
        return score + max(0.0, cls._score(item))

    @classmethod
    def _pdf_list_focus_page(
        cls,
        results: Sequence[Dict[str, Any]],
        query: str = "",
    ) -> int | None:
        pdf_text = [
            item
            for item in results
            if cls._source_kind(item) == "pdf" and cls._chunk_type(item) == "text"
        ]
        if not pdf_text:
            return None

        subject_terms = cls._pdf_query_subject_terms(query)
        best_item: Dict[str, Any] | None = None
        best_score = float("-inf")

        for item in pdf_text:
            content = str(item.get("content", "") or "")
            lower = content.lower()
            lexical = sum(3.0 for term in subject_terms if term in lower)
            heading_bonus = 0.0
            if any(
                marker in lower
                for marker in (
                    "key market drivers", "key drivers", "market drivers",
                    "key factors", "factors", "drivers", "types of",
                    "categories of", "components of",
                )
            ):
                heading_bonus += 4.0
            if re.search(r"(?<!\d)1\s*[\).]\s*[A-Za-z]", content):
                heading_bonus += 2.0
            score = lexical + heading_bonus + cls._score(item)
            if score > best_score:
                best_score = score
                best_item = item

        return cls._page_number(best_item) if best_item is not None else None

    @staticmethod
    def _explicit_page_number(query: str) -> int | None:
        match = re.search(r"\bpage\s*(\d+)\b", str(query or ""), re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _page_number(item: Dict[str, Any]) -> int | None:
        try:
            return int((item.get("metadata") or {}).get("page_number"))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _select_txt_coverage(
        cls,
        *,
        results: Sequence[Dict[str, Any]],
        query: str,
        top_k: int,
    ) -> list[Dict[str, Any]]:
        """
        Re-rank only TXT candidates using the semantic score already returned
        by Chroma plus lightweight lexical/number coverage. No second embedding
        or vector-search call is made, which keeps TXT retrieval fast.
        """
        if not results:
            return []

        query_tokens = cls._lexical_tokens(query)
        query_numbers = cls._query_numbers(query)

        def hybrid(item: Dict[str, Any]) -> tuple[float, float, float]:
            content = str(item.get("content", "") or "")
            content_tokens = cls._lexical_tokens(content)
            content_numbers = cls._query_numbers(content)

            token_overlap = (
                len(query_tokens & content_tokens) / max(1, len(query_tokens))
            )
            number_overlap = (
                len(query_numbers & content_numbers) / max(1, len(query_numbers))
                if query_numbers else 0.0
            )
            semantic = cls._score(item)
            combined = semantic + (0.18 * token_overlap) + (0.28 * number_overlap)
            return combined, number_overlap, token_overlap

        ranked = sorted(list(results), key=hybrid, reverse=True)
        selected = ranked[:top_k]

        # If the user quoted source values in the question (common for
        # cross-section TXT arithmetic), keep the smallest set of retrieved
        # chunks needed to cover those values.
        if query_numbers:
            required: list[Dict[str, Any]] = []
            uncovered = set(query_numbers)
            for item in ranked:
                content_numbers = cls._query_numbers(
                    str(item.get("content", "") or "")
                )
                covered = uncovered & content_numbers
                if covered:
                    required.append(item)
                    uncovered -= covered
                if not uncovered:
                    break
            selected = cls._ensure_required(
                selected=selected,
                required=required,
                top_k=top_k,
            )

        return selected[:top_k]

    @staticmethod
    def _lexical_tokens(text: str) -> set[str]:
        stop = {
            "the", "and", "for", "from", "with", "what", "which", "were",
            "was", "that", "this", "how", "many", "much", "both", "give",
            "into", "than", "their", "its", "had", "has", "have", "does",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", str(text or "").lower())
            if token not in stop
        }

    @staticmethod
    def _query_numbers(text: str) -> set[str]:
        values: set[str] = set()
        for raw in re.findall(
            r"(?<!\w)[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?",
            str(text or ""),
        ):
            normalized = raw.replace(",", "").replace("%", "").strip()
            if normalized:
                values.add(normalized)
        return values

    @staticmethod
    def _requested_modality_match(
        item: Dict[str, Any],
        requested: set[str],
    ) -> int:
        if not requested:
            return 0
        chunk_type = str(
            (item.get("metadata") or {}).get("chunk_type", "")
        ).lower()
        return 1 if chunk_type in requested else 0

    @staticmethod
    def _best_candidate(
        results: Sequence[Dict[str, Any]],
        predicate,
    ) -> Dict[str, Any] | None:
        candidates = [item for item in results if predicate(item)]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: float(item.get("relevance_score", 0.0) or 0.0),
        )

    @staticmethod
    def _has_visual_asset_metadata(item: Dict[str, Any]) -> bool:
        metadata = item.get("metadata") or {}
        for key in ("image_path", "source_path", "path", "source_location"):
            value = metadata.get(key)
            if not value:
                continue
            suffix = Path(str(value)).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                return True
        return False

    @staticmethod
    def _ensure_required(
        *,
        selected: list[Dict[str, Any]],
        required: Sequence[Dict[str, Any]],
        top_k: int,
    ) -> list[Dict[str, Any]]:
        result = list(selected)
        required_ids = {
            str(item.get("chunk_id", ""))
            for item in required
            if str(item.get("chunk_id", ""))
        }

        for required_item in required:
            required_id = str(required_item.get("chunk_id", ""))
            if not required_id:
                continue

            if any(str(item.get("chunk_id", "")) == required_id for item in result):
                continue

            if len(result) < top_k:
                result.append(required_item)
                continue

            replace_index = None
            for index in range(len(result) - 1, -1, -1):
                current_id = str(result[index].get("chunk_id", ""))
                if current_id not in required_ids:
                    replace_index = index
                    break

            if replace_index is None:
                replace_index = len(result) - 1

            result[replace_index] = required_item

        return result

    def _retrieve_pdf_page_text_chunks(
        self,
        *,
        filters: Dict[str, Any],
        page_numbers: Sequence[int],
    ) -> List[Dict[str, Any]]:
        """
        Return every already-indexed PDF text chunk from a small authenticated
        page neighborhood. This is used only for explicit PDF list/detail
        questions so complete numbered lists are not lost to semantic top-k.

        No embedding or model call is performed here. User/document filters are
        preserved exactly, and failure falls back to normal semantic retrieval.
        """
        pages = sorted({int(value) for value in page_numbers if int(value) >= 1})
        if not pages:
            return []

        page_filters = dict(filters)
        page_filters["page_number"] = {"$in": pages}
        page_filters["chunk_type"] = "text"
        where = self._build_chroma_where(page_filters)

        try:
            collection = self.search.chroma.get_collection()
            result = collection.get(
                where=where,
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        ids = list(result.get("ids") or [])
        documents = list(result.get("documents") or [])
        metadatas = list(result.get("metadatas") or [])

        output: list[Dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            content = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            if not str(content or "").strip():
                continue
            output.append(
                {
                    "chunk_id": str(chunk_id),
                    "content": str(content or ""),
                    "metadata": dict(metadata or {}),
                    # Exact authenticated page membership is the routing
                    # signal for this narrow coverage fetch.
                    "distance": 0.0,
                    "relevance_score": 1.0,
                }
            )

        output.sort(
            key=lambda item: (
                self._page_number(item) or 10**9,
                int((item.get("metadata") or {}).get("chunk_index") or 0),
                str(item.get("chunk_id") or ""),
            )
        )
        return output

    def _retrieve_chunk_type(
        self,
        *,
        query_embedding: Sequence[float],
        filters: Dict[str, Any],
        chunk_type: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        chunk_filters = dict(filters)
        chunk_filters["chunk_type"] = chunk_type

        return self.search.search(
            query_embedding=query_embedding,
            top_k=max(1, int(top_k)),
            metadata_filters=self._build_chroma_where(chunk_filters),
        )

    # =========================================================
    # MODALITY RETRIEVAL (frozen helper)
    # =========================================================

    def _retrieve_modalities(
        self,
        *,
        query_embedding: Sequence[float],
        filters: Dict[str, Any],
        modalities: Sequence[str],
        top_k: int,
        min_relevance: float,
    ) -> List[Dict[str, Any]]:
        all_results: Dict[str, Dict[str, Any]] = {}

        for modality in modalities:
            modality_filters = dict(filters)
            modality_filters["chunk_type"] = modality

            results = self.search.search(
                query_embedding=query_embedding,
                top_k=top_k,
                metadata_filters=self._build_chroma_where(modality_filters),
            )

            for result in results:
                score = result.get("relevance_score", 0.0)
                if score < min_relevance:
                    continue

                chunk_id = result.get("chunk_id")
                if not chunk_id:
                    continue

                previous = all_results.get(chunk_id)
                if previous is None or score > previous.get("relevance_score", 0.0):
                    all_results[chunk_id] = result

        ranked = sorted(
            all_results.values(),
            key=lambda item: item.get("relevance_score", 0.0),
            reverse=True,
        )

        return ranked[:top_k]

    # =========================================================
    # CHROMA WHERE BUILDER
    # =========================================================

    @staticmethod
    def _build_chroma_where(filters: Dict[str, Any]) -> Dict[str, Any]:
        if not filters:
            return {}

        conditions: List[Dict[str, Any]] = []

        for key, value in filters.items():
            if value is None:
                continue
            conditions.append({key: value})

        if not conditions:
            return {}

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}
