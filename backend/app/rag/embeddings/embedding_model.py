from __future__ import annotations

from typing import Any, List, Sequence

import os
import time

from app.core.logging import get_logger, log_event
from app.core.performance import EMBEDDING_BATCH_SIZE

logger = get_logger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class EmbeddingModel:
    """
    Local Ollama embedding service used by DocMindAI.

    Phase 14 upload-readiness optimization:
        * keep the embedding model warm in Ollama
        * use the native batch endpoint when available
        * keep a compatibility fallback for older Ollama clients
        * preserve the existing ``embed``/``embed_batch`` public API

    The target model remains the frozen baseline model:
        nomic-embed-text
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        keep_alive: str | None = None,
    ):
        self.model_name = (
            model_name
            or os.getenv(
                "EMBEDDING_MODEL",
                "nomic-embed-text",
            )
        )

        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            )
        ).rstrip("/")

        # Keeping the small embedding model resident removes the
        # repeated Ollama cold-start penalty between uploads.
        self.keep_alive = (
            keep_alive
            or os.getenv(
                "EMBEDDING_KEEP_ALIVE",
                "30m",
            )
        )

        self._client = None

    # =========================================================
    # CLIENT
    # =========================================================

    def _get_client(self):
        if self._client is None:
            try:
                import ollama

                self._client = ollama.Client(
                    host=self.base_url
                )

            except ImportError as exc:
                raise EmbeddingError(
                    "The 'ollama' Python package is not installed."
                ) from exc

        return self._client

    # =========================================================
    # RESPONSE NORMALIZATION
    # =========================================================

    @staticmethod
    def _response_value(
        response: Any,
        key: str,
    ) -> Any:
        if isinstance(response, dict):
            return response.get(key)

        return getattr(
            response,
            key,
            None,
        )

    @classmethod
    def _extract_batch_vectors(
        cls,
        response: Any,
    ) -> list[list[float]] | None:
        vectors = cls._response_value(
            response,
            "embeddings",
        )

        if not vectors:
            return None

        return [
            list(vector)
            for vector in vectors
        ]

    @classmethod
    def _extract_single_vector(
        cls,
        response: Any,
    ) -> list[float] | None:
        # New Ollama /api/embed response.
        vectors = cls._extract_batch_vectors(
            response
        )

        if vectors:
            return vectors[0]

        # Older /api/embeddings response.
        vector = cls._response_value(
            response,
            "embedding",
        )

        if not vector:
            return None

        return list(vector)

    # =========================================================
    # NATIVE EMBED CALL
    # =========================================================

    def _native_embed(
        self,
        inputs: str | Sequence[str],
    ) -> Any:
        """
        Call the modern Ollama ``embed`` API.

        Some older Python client versions do not accept
        ``keep_alive``.  Retry without that keyword before falling
        back to the legacy single-text API.
        """

        client = self._get_client()
        method = getattr(
            client,
            "embed",
            None,
        )

        if not callable(method):
            raise AttributeError(
                "Installed Ollama client has no embed() method."
            )

        try:
            return method(
                model=self.model_name,
                input=inputs,
                keep_alive=self.keep_alive,
            )

        except TypeError:
            # Compatibility with older Ollama Python clients.
            return method(
                model=self.model_name,
                input=inputs,
            )

    # =========================================================
    # SINGLE EMBEDDING
    # =========================================================

    def embed(
        self,
        text: str,
    ) -> List[float]:
        """Generate one embedding while preserving the API."""

        if not text or not text.strip():
            raise ValueError(
                "Cannot generate an embedding for empty text."
            )

        cleaned = text.strip()
        client = self._get_client()

        # Prefer the modern endpoint because it also honours the
        # same warm-model path used by batch ingestion.
        try:
            response = self._native_embed(
                cleaned
            )

            embedding = (
                self._extract_single_vector(
                    response
                )
            )

            if embedding:
                return embedding

        except Exception:
            # Preserve the frozen legacy path below.
            pass

        try:
            legacy_method = getattr(
                client,
                "embeddings",
                None,
            )

            if not callable(legacy_method):
                raise EmbeddingError(
                    "Ollama client exposes neither embed() nor embeddings()."
                )

            response = legacy_method(
                model=self.model_name,
                prompt=cleaned,
            )

            embedding = (
                self._extract_single_vector(
                    response
                )
            )

            if not embedding:
                raise EmbeddingError(
                    "Ollama returned an empty embedding."
                )

            return embedding

        except Exception as exc:
            raise EmbeddingError(
                f"Embedding generation failed for "
                f"model '{self.model_name}': {exc}"
            ) from exc

    # =========================================================
    # TRUE BATCH EMBEDDING
    # =========================================================

    def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> List[List[float]]:
        """
        Generate embeddings using real Ollama request batching.

        For an XLSX producing 8 retrieval chunks and the default
        batch size of 8, the preferred path performs one Ollama
        embedding request rather than eight separate requests.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        cleaned = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned:
            return []

        embeddings: List[List[float]] = []
        started = time.perf_counter()
        native_batch_requests = 0
        fallback_single_requests = 0

        for start in range(
            0,
            len(cleaned),
            batch_size,
        ):
            batch = cleaned[
                start:
                start + batch_size
            ]

            try:
                response = self._native_embed(
                    batch
                )

                batch_vectors = (
                    self._extract_batch_vectors(
                        response
                    )
                )

                if (
                    batch_vectors
                    and len(batch_vectors)
                    == len(batch)
                ):
                    embeddings.extend(
                        batch_vectors
                    )
                    native_batch_requests += 1
                    continue

            except Exception:
                # Older server/client: preserve compatible path.
                pass

            for text in batch:
                embeddings.append(
                    self.embed(text)
                )
                fallback_single_requests += 1

        if len(embeddings) != len(cleaned):
            raise EmbeddingError(
                "Embedding count mismatch. "
                f"Expected {len(cleaned)}, got {len(embeddings)}."
            )

        log_event(
            logger,
            "embedding",
            count=len(embeddings),
            batch_size=batch_size,
            native_batch_requests=(
                native_batch_requests
            ),
            fallback_single_requests=(
                fallback_single_requests
            ),
            latency_ms=round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2,
            ),
        )

        return embeddings

    # =========================================================
    # WARM-UP
    # =========================================================

    def warmup(self) -> bool:
        """
        Load the embedding model into Ollama before the first upload.

        The returned vector is intentionally discarded.  This does
        not change any stored RAG vector or application behaviour.
        """

        started = time.perf_counter()

        try:
            vector = self.embed(
                "DocMindAI embedding warmup"
            )

            ok = bool(vector)

            log_event(
                logger,
                "embedding_warmup",
                model=self.model_name,
                success=ok,
                latency_ms=round(
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000,
                    2,
                ),
            )

            return ok

        except Exception as exc:
            log_event(
                logger,
                "embedding_warmup_failed",
                level=30,
                model=self.model_name,
                error_type=(
                    type(exc).__name__
                ),
                error=str(exc),
            )

            return False

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def dimension(self) -> int | None:
        """Dimension remains determined lazily by Ollama."""
        return None


# Backward-compatible aliases.
EmbeddingService = EmbeddingModel
