"""A scripted LLM for tests: no network, fully deterministic."""

import json
from collections.abc import Callable
from typing import Any

from app.llm.base import ChatMessage, Completion, LLMProvider, LLMUnavailableError

SummaryFn = Callable[[str], str]  # receives the summary prompt's user message


class FakeLLM(LLMProvider):
    name = "fake"
    model = "fake-1"

    def __init__(
        self,
        plans: list[dict[str, Any] | str] | None = None,
        summary: str | SummaryFn = "Here is the result.",
        unavailable: bool = False,
    ) -> None:
        self.plans = list(plans or [])
        self.summary = summary
        self.unavailable = unavailable
        self.calls: list[tuple[list[ChatMessage], dict[str, Any] | None]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
    ) -> Completion:
        if self.unavailable:
            raise LLMUnavailableError("The fake model is switched off.")
        self.calls.append((messages, json_schema))
        if json_schema is not None:
            if not self.plans:
                raise AssertionError("FakeLLM ran out of scripted plans")
            nxt = self.plans.pop(0)
            text = nxt if isinstance(nxt, str) else json.dumps(nxt)
        else:
            text = self.summary(messages[-1].content) if callable(self.summary) else self.summary
        return Completion(text, prompt_tokens=100, completion_tokens=20, latency_ms=5)

    @property
    def plan_calls(self) -> list[list[ChatMessage]]:
        return [m for m, schema in self.calls if schema is not None]
