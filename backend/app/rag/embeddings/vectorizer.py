from __future__ import annotations

from typing import Any, List, Sequence

from .embedding_model import EmbeddingModel


class Vectorizer:
    """
    Converts retrieval chunks into vectors.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        batch_size: int = 16,
    ):
        self.embedding_model = (
            embedding_model
            or EmbeddingModel()
        )

        self.batch_size = batch_size

    def embed_text(
        self,
        text: str,
    ) -> List[float]:

        return self.embedding_model.embed(text)

    def embed_chunks(
        self,
        chunks: Sequence[Any],
    ) -> List[List[float]]:
        """
        Generate vectors from Phase 7 RetrievalChunk objects.
        """

        texts = []

        for chunk in chunks:
            content = getattr(
                chunk,
                "content",
                None,
            )

            if not content:
                continue

            texts.append(content)

        return self.embedding_model.embed_batch(
            texts,
            batch_size=self.batch_size,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:

        return self.embedding_model.embed_batch(
            texts,
            batch_size=self.batch_size,
        )