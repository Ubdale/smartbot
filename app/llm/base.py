"""Provider-agnostic LLM interface.

Concrete providers implement only `complete()` (one chat call, optionally
constrained to a JSON schema). `plan()` and `summarize()` are shared, so every
provider gets the same prompts and the same validation.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from app.llm import prompts
from app.schemas.query_plan import QueryPlan

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


@dataclass(frozen=True)
class LLMCall:
    """One logged call, for the usage table."""

    purpose: Literal["plan", "summary"]
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    success: bool


@dataclass(frozen=True)
class PlanAttempt:
    raw: str
    plan: QueryPlan | None
    error: str | None  # JSON/schema problem, phrased so the model can fix it
    call: LLMCall


class LLMUnavailableError(RuntimeError):
    """The model can't be reached. The message is safe to show users."""


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
    ) -> Completion:
        """Run one chat completion. Raise LLMUnavailableError if the model can't be reached."""

    # --- shared behaviour -------------------------------------------------------------

    def plan(
        self,
        question: str,
        schema_text: str,
        history: list[ChatMessage],
        feedback: tuple[str, str] | None = None,
    ) -> PlanAttempt:
        """Ask for a QueryPlan. `feedback` = (previous raw reply, what was wrong with it)."""
        messages = prompts.plan_messages(question, schema_text, history)
        if feedback:
            bad_reply, problem = feedback
            messages += [
                ChatMessage("assistant", bad_reply),
                ChatMessage("user", prompts.retry_instruction(problem)),
            ]
        started = time.perf_counter()
        try:
            out = self.complete(messages, json_schema=prompts.plan_json_schema(), max_tokens=600)
        except LLMUnavailableError:
            raise
        call = self._call("plan", out, started, success=True)
        try:
            plan = QueryPlan.model_validate_json(out.text)
        except ValidationError as exc:
            problem = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'plan'}: {e['msg']}" for e in exc.errors()
            )
            return PlanAttempt(out.text, None, problem[:800], call)
        return PlanAttempt(out.text, plan, None, call)

    def summarize(self, question: str, description: str, table_text: str) -> tuple[str, LLMCall]:
        messages = prompts.summary_messages(question, description, table_text)
        started = time.perf_counter()
        out = self.complete(messages, max_tokens=220)
        return out.text.strip(), self._call("summary", out, started, success=True)

    def _call(self, purpose: Literal["plan", "summary"], out: Completion, started: float, *,
              success: bool) -> LLMCall:  # fmt: skip
        latency = out.latency_ms or int((time.perf_counter() - started) * 1000)
        return LLMCall(
            purpose, self.name, self.model, out.prompt_tokens, out.completion_tokens, latency,
            success,
        )  # fmt: skip
