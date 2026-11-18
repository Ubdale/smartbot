"""Local LLM via Ollama (free, unlimited). Structured output uses Ollama's `format` schema."""

import time
from typing import Any

import httpx

from app.llm.base import ChatMessage, Completion, LLMProvider, LLMUnavailableError

TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, base_url: str, client: httpx.Client | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=TIMEOUT)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        if json_schema is not None:
            payload["format"] = json_schema

        started = time.perf_counter()
        try:
            resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Can't reach Ollama at {self.base_url}. Is it running (`ollama serve`)?"
            ) from exc
        if resp.status_code == 404:
            raise LLMUnavailableError(
                f"Ollama doesn't have the model {self.model!r}. Run: ollama pull {self.model}"
            )
        if resp.status_code >= 400:
            raise LLMUnavailableError(f"Ollama returned an error ({resp.status_code}).")

        data = resp.json()
        return Completion(
            text=str(data.get("message", {}).get("content", "")),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
