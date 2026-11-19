"""A small in-memory per-user limit on questions (protects free LLM quotas and the
self-hosted model from one runaway user). One server process, so memory is enough;
with several instances, move this to Redis."""

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, status


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window: float = 60.0) -> None:
        """Raise 429 if `key` already made `limit` requests in the last `window` seconds."""
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= window:
                hits.popleft()
            if len(hits) >= limit:
                wait = int(window - (now - hits[0])) + 1
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    f"You're asking very quickly. Please wait {wait} seconds and try again.",
                    headers={"Retry-After": str(wait)},
                )
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


questions = RateLimiter()
