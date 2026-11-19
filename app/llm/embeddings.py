"""Embeddings: text -> vectors, so search can match by meaning, not just by words.

Runs on the same self-hosted Ollama server as the chat model (nomic-embed-text,
~270 MB, CPU friendly). Always optional: when it's unavailable, search falls back to
keywords, so an answer never fails because of embeddings.
"""

import threading
import time
from typing import TYPE_CHECKING

import httpx
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from app.core.config import Settings

Vectors = npt.NDArray[np.float32]
BATCH = 64
MAX_CHARS = 2000  # per text; nomic-embed-text reads ~2k tokens
COOLDOWN_SECONDS = 300  # after a failure, don't keep trying (and slowing answers down)


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    """nomic-embed-text over Ollama's /api/embed. Vectors are L2-normalised, so a dot
    product is the cosine similarity."""

    name = "ollama"

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout_seconds: float = 120.0,
                 client: httpx.Client | None = None) -> None:  # fmt: skip
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        timeout = httpx.Timeout(connect=5.0, read=timeout_seconds, write=30.0, pool=5.0)
        self._client = client or httpx.Client(timeout=timeout, headers=headers)
        self._failed_at = 0.0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return time.monotonic() - self._failed_at > COOLDOWN_SECONDS

    def embed(self, texts: list[str], *, query: bool = False) -> Vectors:
        """Raises EmbeddingError (and pauses embeddings for a few minutes) on failure."""
        if not self.available:
            raise EmbeddingError("Embeddings are paused after a recent failure.")
        # nomic-embed-text was trained with these task prefixes.
        prefix = "search_query: " if query else "search_document: "
        out: list[list[float]] = []
        try:
            for i in range(0, len(texts), BATCH):
                batch = [prefix + (t or " ")[:MAX_CHARS] for t in texts[i : i + BATCH]]
                resp = self._client.post(f"{self.base_url}/api/embed",
                                         json={"model": self.model, "input": batch,
                                               "truncate": True})  # fmt: skip
                resp.raise_for_status()
                vectors = resp.json().get("embeddings") or []
                if len(vectors) != len(batch):
                    raise EmbeddingError("The embedding server returned the wrong count.")
                out += vectors
        except (httpx.HTTPError, ValueError, EmbeddingError) as exc:
            with self._lock:
                self._failed_at = time.monotonic()
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        matrix = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalised: Vectors = (matrix / np.maximum(norms, 1e-12)).astype(np.float32)
        return normalised


_embedders: dict[tuple[str, str], Embedder] = {}
_embedders_lock = threading.Lock()


def get_embedder(settings: "Settings") -> Embedder | None:
    """The configured embedder, or None (then search uses keywords only)."""
    provider = settings.embed_provider
    if provider == "auto":
        provider = "ollama" if settings.llm_provider == "ollama" and settings.app_env != "test" \
            else "none"  # fmt: skip
    if provider != "ollama":
        return None
    key = (settings.ollama_base_url, settings.embed_model)
    with _embedders_lock:
        if key not in _embedders:
            api_key = settings.ollama_api_key.get_secret_value() if settings.ollama_api_key \
                else None  # fmt: skip
            _embedders[key] = Embedder(settings.ollama_base_url, settings.embed_model, api_key)
        return _embedders[key]
