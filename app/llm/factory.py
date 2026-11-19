"""Pick the LLM provider(s) from settings. Switching providers needs no code changes.

LLM_PROVIDER is the main model; LLM_FALLBACK_PROVIDER (optional) takes over for a
request whenever the main one is unavailable or out of free-tier quota.
"""

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status

from app.core.config import DEFAULT_MODELS, DEFAULT_SUMMARY_MODELS, Settings, get_settings
from app.llm.base import (
    ChatMessage,
    Completion,
    LLMCall,
    LLMProvider,
    LLMUnavailableError,
    PlanAttempt,
)
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import OllamaProvider


class FallbackProvider(LLMProvider):
    """Try each provider in order; move on when one is unavailable or rate-limited."""

    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = providers
        self.name = "+".join(p.name for p in providers)
        self.model = providers[0].model

    def _first_available[T](self, call: Callable[[LLMProvider], T]) -> T:
        error = LLMUnavailableError("No language model is available right now.")
        for provider in self.providers:
            try:
                return call(provider)
            except LLMUnavailableError as exc:
                error = exc
        raise error

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
    ) -> Completion:
        return self._first_available(
            lambda p: p.complete(messages, json_schema=json_schema, max_tokens=max_tokens)
        )

    # plan/summarize delegate whole calls so usage is logged under the provider that answered.
    def plan(
        self,
        question: str,
        schema_text: str,
        history: list[ChatMessage],
        feedback: tuple[str, str] | None = None,
    ) -> PlanAttempt:
        return self._first_available(lambda p: p.plan(question, schema_text, history, feedback))

    def summarize(self, question: str, description: str, table_text: str) -> tuple[str, LLMCall]:
        return self._first_available(lambda p: p.summarize(question, description, table_text))


# Seconds to wait for a self-hosted model before switching to the backup provider.
OLLAMA_TIMEOUT_WITH_FALLBACK = {False: 60.0, True: 30.0}  # {summary?: seconds}


def _check_model_family(provider: str, model: str) -> None:
    """Catch a leftover LLM_MODEL from another provider (e.g. an Ollama tag on Gemini)."""
    wrong = (provider == "gemini" and not model.startswith("gemini")) or (
        provider != "ollama" and ":" in model  # "name:tag" is Ollama's format
    )
    if wrong:
        raise LLMUnavailableError(
            f"LLM_MODEL={model!r} doesn't look like a {provider} model. Leave LLM_MODEL empty "
            f"to use {DEFAULT_MODELS[provider]!r}."
        )


def _model_for(settings: Settings, provider: str, summary: bool) -> str:
    if provider == settings.llm_provider:
        return settings.summary_model_name if summary else settings.model_name
    if summary:  # the fallback uses its provider's defaults
        return DEFAULT_SUMMARY_MODELS.get(provider, DEFAULT_MODELS[provider])
    return DEFAULT_MODELS[provider]


def _single(
    settings: Settings, provider: str, *, summary: bool, patient: bool, has_backup: bool = False
) -> LLMProvider:
    model = _model_for(settings, provider, summary)
    _check_model_family(provider, model)
    if provider == "ollama":
        timeout = settings.ollama_timeout_seconds
        if has_backup:
            # A backup is waiting: give up early so the whole request (plan + summary,
            # possibly via the backup) fits Vercel's 120 s proxy limit.
            timeout = min(timeout, OLLAMA_TIMEOUT_WITH_FALLBACK[summary])
        return OllamaProvider(
            model,
            settings.ollama_base_url,
            num_gpu=settings.ollama_num_gpu,
            timeout_seconds=timeout,
            api_key=settings.ollama_api_key.get_secret_value() if settings.ollama_api_key else None,
        )
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMUnavailableError("Gemini needs GEMINI_API_KEY.")
        return GeminiProvider(
            model,
            settings.gemini_api_key.get_secret_value(),
            # Wait out a short rate limit only when there's nothing else to try.
            max_retry_wait=20 if patient else 0,
        )
    if not settings.groq_api_key:
        raise LLMUnavailableError("Groq needs GROQ_API_KEY.")
    return GroqProvider(model, settings.groq_api_key.get_secret_value())


def make_provider(settings: Settings, *, summary: bool = False) -> LLMProvider:
    fallback = settings.llm_fallback_provider
    if fallback is None or fallback == settings.llm_provider:
        # Summaries never wait on a rate limit: a code-written summary is shown instead.
        return _single(settings, settings.llm_provider, summary=summary, patient=not summary)
    return FallbackProvider(
        [
            _single(
                settings, settings.llm_provider, summary=summary, patient=False, has_backup=True
            ),
            _single(settings, fallback, summary=summary, patient=not summary),
        ]
    )


_cached: dict[tuple[str, ...], LLMProvider] = {}


def _cache_key(settings: Settings, summary: bool) -> tuple[str, ...]:
    return (
        settings.llm_provider,
        str(settings.llm_fallback_provider),
        settings.summary_model_name if summary else settings.model_name,
        str(summary),
        settings.ollama_base_url,
        str(settings.ollama_num_gpu),
    )


def _build(settings: Settings, summary: bool) -> LLMProvider:
    key = _cache_key(settings, summary)
    if key not in _cached:
        try:
            _cached[key] = make_provider(settings, summary=summary)
        except LLMUnavailableError as exc:  # misconfiguration: say what's wrong, not "500"
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return _cached[key]


def get_llm(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    return _build(settings, summary=False)


def get_summary_llm(
    settings: Annotated[Settings, Depends(get_settings)],
    planner: Annotated[LLMProvider, Depends(get_llm)],
) -> LLMProvider:
    """The model that writes summaries; the planning model unless a separate one is set."""
    if settings.summary_model_name == settings.model_name and not settings.llm_fallback_provider:
        return planner
    return _build(settings, summary=True)
