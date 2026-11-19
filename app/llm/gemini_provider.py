"""Google Gemini (free tier) over its REST API. Structured output via responseJsonSchema."""

import base64
import re
import time
from typing import Any

import httpx

from app.llm.base import ChatMessage, Completion, LLMProvider, LLMUnavailableError

API = "https://generativelanguage.googleapis.com/v1beta"
# Free tier: 5 requests/minute per model, with retry hints of ~10-20 s. Waiting that long
# beats failing for a question; summaries pass 0 and fall back to a code summary instead.
MAX_RETRY_WAIT = 20.0


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        model: str,
        api_key: str,
        client: httpx.Client | None = None,
        *,
        timeout_seconds: float = 60,
        max_retry_wait: float = MAX_RETRY_WAIT,
        sleep: Any = time.sleep,
    ) -> None:
        self.model = model
        self.max_retry_wait = max_retry_wait
        self._key = api_key
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_seconds, connect=10))
        self._sleep = sleep

    def _body(
        self, messages: list[ChatMessage], json_schema: dict[str, Any] | None, max_tokens: int
    ) -> dict[str, Any]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        config: dict[str, Any] = {"temperature": 0, "maxOutputTokens": max_tokens}
        if json_schema is not None:
            config["responseMimeType"] = "application/json"
            config["responseJsonSchema"] = json_schema
        if self.model.startswith("gemini-2.5"):
            # 2.5 models "think" by default: slower and spends output tokens. Our tasks are
            # small and fully specified, so skip it.
            config["thinkingConfig"] = {"thinkingBudget": 0}
        body: dict[str, Any] = {"contents": contents, "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
    ) -> Completion:
        return self._send(self._body(messages, json_schema, max_tokens))

    def complete_with_image(
        self,
        prompt: str,
        image: bytes,
        mime: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> Completion:
        body = self._body([ChatMessage("user", prompt)], json_schema, max_tokens)
        image_part = {"inline_data": {"mime_type": mime, "data": base64.b64encode(image).decode()}}
        body["contents"][0]["parts"].insert(0, image_part)
        return self._send(body)

    def _send(self, body: dict[str, Any]) -> Completion:
        url = f"{API}/models/{self.model}:generateContent"
        started = time.perf_counter()

        for attempt in range(2):
            try:
                resp = self._client.post(url, json=body, headers={"x-goog-api-key": self._key})
            except httpx.TimeoutException as exc:
                raise LLMUnavailableError("Gemini took too long to answer. Try again.") from exc
            except httpx.HTTPError as exc:
                raise LLMUnavailableError("Can't reach the Gemini API right now.") from exc

            if resp.status_code in (429, 503):
                # 429 = free-tier rate limit, 503 = model overloaded: both are temporary.
                wait = _retry_delay(resp) if resp.status_code == 429 else 2.0
                if attempt == 0 and wait <= self.max_retry_wait:
                    self._sleep(wait)
                    continue
                raise LLMUnavailableError(
                    "The free Gemini quota is used up for the moment. Please try again in a minute."
                    if resp.status_code == 429
                    else "Gemini is overloaded right now. Please try again in a moment."
                )
            break

        if resp.status_code in (401, 403):
            raise LLMUnavailableError("Gemini rejected the API key. Check GEMINI_API_KEY.")
        if resp.status_code == 404:
            raise LLMUnavailableError(f"Gemini has no model {self.model!r}. Check LLM_MODEL.")
        if resp.status_code >= 400:
            raise LLMUnavailableError(f"Gemini returned an error ({resp.status_code}).")

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason", "no answer")
            raise LLMUnavailableError(f"Gemini didn't answer ({reason}). Try rephrasing.")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata", {})
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            completion_tokens=int(usage.get("candidatesTokenCount") or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def _retry_delay(resp: httpx.Response) -> float:
    """Seconds to wait, from Gemini's RetryInfo ("7s") or the Retry-After header."""
    try:
        for detail in resp.json().get("error", {}).get("details", []):
            if m := re.fullmatch(r"(\d+(?:\.\d+)?)s", str(detail.get("retryDelay", ""))):
                return float(m[1])
    except ValueError:
        pass
    header = resp.headers.get("retry-after", "")
    return float(header) if header.replace(".", "", 1).isdigit() else 60.0
