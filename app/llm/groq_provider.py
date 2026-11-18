"""Groq (free tier) via its OpenAI-compatible chat API, in JSON mode.

JSON mode guarantees valid JSON but not our exact schema, so the schema is added to
the system prompt; the planner's validation + one retry handles the rest.
"""

import json
import time
from typing import Any

import httpx

from app.llm.base import ChatMessage, Completion, LLMProvider, LLMUnavailableError

API = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(
        self,
        model: str,
        api_key: str,
        client: httpx.Client | None = None,
        *,
        timeout_seconds: float = 60,
    ) -> None:
        self.model = model
        self._key = api_key
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_seconds, connect=10))

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
    ) -> Completion:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            body["response_format"] = {"type": "json_object"}
            hint = "Reply with one JSON object matching this JSON schema:\n" + json.dumps(
                json_schema, separators=(",", ":")
            )
            if msgs and msgs[0]["role"] == "system":
                msgs[0] = {"role": "system", "content": f"{msgs[0]['content']}\n\n{hint}"}
            else:
                msgs.insert(0, {"role": "system", "content": hint})

        started = time.perf_counter()
        try:
            resp = self._client.post(
                API, json=body, headers={"Authorization": f"Bearer {self._key}"}
            )
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("Groq took too long to answer. Try again.") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError("Can't reach the Groq API right now.") from exc

        if resp.status_code == 429:
            raise LLMUnavailableError(
                "The free Groq quota is used up for the moment. Please try again in a minute."
            )
        if resp.status_code in (401, 403):
            raise LLMUnavailableError("Groq rejected the API key. Check GROQ_API_KEY.")
        if resp.status_code >= 400:
            raise LLMUnavailableError(f"Groq returned an error ({resp.status_code}).")

        data = resp.json()
        usage = data.get("usage", {})
        choices = data.get("choices") or [{}]
        return Completion(
            text=str(choices[0].get("message", {}).get("content") or ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
