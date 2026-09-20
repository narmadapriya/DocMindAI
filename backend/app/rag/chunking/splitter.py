from __future__ import annotations

import re
from typing import List


class TextSplitter:
    """
    Lightweight semantic-aware text splitter.

    The splitter avoids blindly splitting in the middle of
    sentences when possible and applies configurable overlap.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        min_chunk_size: int = 40,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def split(self, text: str) -> List[str]:
        """
        Split text into retrieval-friendly chunks.
        """

        if not text:
            return []

        text = self._normalize(text)

        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = self._paragraphs(text)

        chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:

            if not paragraph:
                continue

            candidate = (
                paragraph
                if not current
                else f"{current}\n\n{paragraph}"
            )

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())

            # Paragraph itself is larger than the target.
            if len(paragraph) > self.chunk_size:
                pieces = self._split_long_text(paragraph)

                if pieces:
                    if chunks:
                        pieces = self._apply_overlap(
                            chunks[-1],
                            pieces,
                        )

                    chunks.extend(pieces[:-1])
                    current = pieces[-1]
                else:
                    current = ""
            else:
                current = paragraph

        if current.strip():
            chunks.append(current.strip())

        return self._merge_tiny_chunks(chunks)

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove excessive horizontal whitespace.
        text = re.sub(r"[ \t]+", " ", text)

        # Keep paragraph boundaries.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def _paragraphs(text: str) -> List[str]:
        return [
            part.strip()
            for part in re.split(r"\n\s*\n", text)
            if part.strip()
        ]

    def _split_long_text(self, text: str) -> List[str]:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        pieces: List[str] = []
        current = ""

        for sentence in sentences:

            if not sentence:
                continue

            candidate = (
                sentence
                if not current
                else f"{current} {sentence}"
            )

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                pieces.append(current.strip())

            # A single sentence is still too large.
            if len(sentence) > self.chunk_size:
                pieces.extend(
                    self._hard_split(sentence)
                )
                current = ""
            else:
                current = sentence

        if current:
            pieces.append(current.strip())

        return pieces

    def _hard_split(self, text: str) -> List[str]:
        pieces = []

        start = 0
        length = len(text)

        while start < length:
            end = min(
                start + self.chunk_size,
                length,
            )

            piece = text[start:end].strip()

            if piece:
                pieces.append(piece)

            if end >= length:
                break

            start = max(
                0,
                end - self.chunk_overlap,
            )

        return pieces

    def _apply_overlap(
        self,
        previous: str,
        pieces: List[str],
    ) -> List[str]:

        if not previous or not pieces:
            return pieces

        overlap = previous[-self.chunk_overlap:]

        if not overlap:
            return pieces

        result = pieces.copy()

        result[0] = (
            overlap.rstrip()
            + " "
            + result[0].lstrip()
        )

        if len(result[0]) > self.chunk_size:
            result[0] = result[0][-self.chunk_size:]

        return result

    def _merge_tiny_chunks(
        self,
        chunks: List[str],
    ) -> List[str]:

        if len(chunks) <= 1:
            return chunks

        result: List[str] = []

        for chunk in chunks:

            if (
                result
                and len(chunk) < self.min_chunk_size
                and len(result[-1]) + len(chunk) + 2
                <= self.chunk_size
            ):
                result[-1] = (
                    result[-1]
                    + "\n\n"
                    + chunk
                )
            else:
                result.append(chunk)

        return result


# Backward-compatible alias.
RecursiveTextSplitter = TextSplitter