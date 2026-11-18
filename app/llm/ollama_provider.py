"""Local LLM via Ollama (free, unlimited). Structured output uses Ollama's `format` schema."""

import contextlib
import time
from typing import Any

import httpx

from app.llm.base import ChatMessage, Completion, LLMProvider, LLMUnavailableError


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str,
        client: httpx.Client | None = None,
        *,
        num_gpu: int | None = None,
        timeout_seconds: float = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_gpu = num_gpu
        timeout = httpx.Timeout(connect=5.0, read=timeout_seconds, write=30.0, pool=5.0)
        self._client = client or httpx.Client(timeout=timeout)

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
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        if json_schema is not None:
            payload["format"] = json_schema

        started = time.perf_counter()
        try:
            resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(
                "The local model took too long to answer. On a CPU-only machine try a smaller "
                "model (e.g. qwen2.5:3b-instruct) or raise OLLAMA_TIMEOUT_SECONDS."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Can't reach Ollama at {self.base_url}. Is it running (`ollama serve`)?"
            ) from exc
        if resp.status_code == 404:
            raise LLMUnavailableError(
                f"Ollama doesn't have the model {self.model!r}. Run: ollama pull {self.model}"
            )
        if resp.status_code >= 400:
            detail = ""
            with contextlib.suppress(ValueError):
                detail = str(resp.json().get("error", ""))[:200]
            hint = (
                " The GPU ran out of memory: set OLLAMA_NUM_GPU=0 to run on the CPU."
                if "out of memory" in detail.lower()
                else ""
            )
            raise LLMUnavailableError(f"Ollama returned an error ({resp.status_code}).{hint}")

        data = resp.json()
        return Completion(
            text=str(data.get("message", {}).get("content", "")),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
