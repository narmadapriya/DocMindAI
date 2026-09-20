from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar

T = TypeVar("T")


def env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


EMBEDDING_BATCH_SIZE = env_int("EMBEDDING_BATCH_SIZE", 8)
RAG_MAX_TOP_K = env_int("RAG_MAX_TOP_K", 10)
RAG_CACHE_MAX_ENTRIES = env_int("RAG_CACHE_MAX_ENTRIES", 128)
RAG_CACHE_TTL_SECONDS = env_int("RAG_CACHE_TTL_SECONDS", 300)
OLLAMA_NUM_PREDICT = env_int("OLLAMA_NUM_PREDICT", 512)
SLOW_REQUEST_MS = env_int("SLOW_REQUEST_MS", 10_000)


@dataclass
class _Item(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Tiny thread-safe bounded TTL cache suitable for an 8-GB local machine."""

    def __init__(self, max_entries: int = RAG_CACHE_MAX_ENTRIES, ttl_seconds: int = RAG_CACHE_TTL_SECONDS):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[Hashable, _Item[T]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: Hashable) -> T | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if item.expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return item.value

    def set(self, key: Hashable, value: T) -> None:
        with self._lock:
            self._items[key] = _Item(value, time.monotonic() + self.ttl_seconds)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
