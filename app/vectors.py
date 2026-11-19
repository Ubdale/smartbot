"""Vectors for a dataset's texts (document passages, a text column's values), computed
once and kept next to the dataset's local Parquet file.

Small sets are embedded on the spot; large ones in the background (started at upload),
and until they're ready search simply uses keywords.
"""

import contextlib
import hashlib
import logging
import threading
from pathlib import Path

import numpy as np

from app.llm.embeddings import Embedder, EmbeddingError, Vectors

log = logging.getLogger(__name__)
SYNC_LIMIT = 300  # up to this many texts: embed now (a few seconds), else in background
MAX_TEXTS = 20_000

_memory: dict[str, Vectors] = {}
_pending: set[str] = set()
_lock = threading.Lock()


def _file(parquet: Path, tag: str, texts: list[str]) -> Path:
    digest = hashlib.sha256("\x1f".join(texts).encode("utf-8", "replace")).hexdigest()[:16]
    safe = "".join(ch if ch.isalnum() else "_" for ch in tag)[:40]
    return parquet.with_name(f"{parquet.stem}.{safe}.{digest}.vec.npy")


def _compute(embedder: Embedder, texts: list[str], path: Path) -> Vectors | None:
    try:
        vectors = embedder.embed(texts)
    except EmbeddingError as exc:
        log.warning("Embedding %d texts failed: %s", len(texts), exc)
        return None
    with contextlib.suppress(OSError):  # if saving fails, memory still has them
        np.save(path, vectors)
    with _lock:
        _memory[str(path)] = vectors
    return vectors


def get_vectors(
    embedder: Embedder | None, parquet: Path, tag: str, texts: list[str], *, wait: bool = False
) -> Vectors | None:
    """Vectors for `texts` (row i = texts[i]), or None if not ready / unavailable.

    `wait`: embed now even if there are many texts (used by background warm-up).
    """
    if embedder is None or not texts or len(texts) > MAX_TEXTS:
        return None
    path = _file(parquet, tag, texts)
    key = str(path)
    with _lock:
        if key in _memory:
            return _memory[key]
        if key in _pending:
            return None
    if path.exists():
        try:
            vectors: Vectors = np.load(path)
            if vectors.shape[0] == len(texts):
                with _lock:
                    _memory[key] = vectors
                return vectors
        except (OSError, ValueError):
            pass
    if not embedder.available:
        return None
    if wait or len(texts) <= SYNC_LIMIT:
        return _compute(embedder, texts, path)
    warm(embedder, parquet, tag, texts)
    return None


def warm(embedder: Embedder | None, parquet: Path, tag: str, texts: list[str]) -> None:
    """Embed in a background thread, so the first question doesn't wait for it."""
    if embedder is None or not texts or len(texts) > MAX_TEXTS:
        return
    path = _file(parquet, tag, texts)
    key = str(path)
    with _lock:
        if key in _memory or key in _pending or path.exists():
            return
        _pending.add(key)

    def run() -> None:
        try:
            _compute(embedder, texts, path)
        finally:
            with _lock:
                _pending.discard(key)

    threading.Thread(target=run, name="embed", daemon=True).start()


def rank(embedder: Embedder, vectors: Vectors, query: str, k: int) -> list[int]:
    """Indices of the k texts closest in meaning to `query`, best first ([] on failure)."""
    try:
        q = embedder.embed([query], query=True)[0]
    except EmbeddingError:
        return []
    scores = vectors @ q
    top = np.argsort(-scores)[:k]
    return [int(i) for i in top]


def fuse(*rankings: list[int], k: int) -> list[int]:
    """Reciprocal rank fusion: items ranked high by either method come first."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking):
            score[item] = score.get(item, 0.0) + 1.0 / (60 + position)
    return sorted(score, key=lambda i: -score[i])[:k]
