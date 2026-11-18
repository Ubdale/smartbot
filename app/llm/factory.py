"""Pick the LLM provider from settings (LLM_PROVIDER / LLM_MODEL). No code changes needed."""

from typing import Annotated

from fastapi import Depends

from app.core.config import DEFAULT_MODELS, Settings, get_settings
from app.llm.base import LLMProvider, LLMUnavailableError
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import OllamaProvider


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


def make_provider(settings: Settings, *, summary: bool = False) -> LLMProvider:
    model = settings.summary_model_name if summary else settings.model_name
    _check_model_family(settings.llm_provider, model)
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            model,
            settings.ollama_base_url,
            num_gpu=settings.ollama_num_gpu,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMUnavailableError("LLM_PROVIDER=gemini needs GEMINI_API_KEY.")
        # Summaries never wait on a rate limit: a code-written summary is shown instead.
        return GeminiProvider(
            model, settings.gemini_api_key.get_secret_value(), max_retry_wait=0 if summary else 20
        )
    if not settings.groq_api_key:
        raise LLMUnavailableError("LLM_PROVIDER=groq needs GROQ_API_KEY.")
    return GroqProvider(model, settings.groq_api_key.get_secret_value())


_cached: dict[tuple[str, ...], LLMProvider] = {}


def _cache_key(settings: Settings, model: str, summary: bool) -> tuple[str, ...]:
    return (
        settings.llm_provider,
        model,
        str(summary),
        settings.ollama_base_url,
        str(settings.ollama_num_gpu),
    )


def get_llm(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    key = _cache_key(settings, settings.model_name, False)
    if key not in _cached:
        _cached[key] = make_provider(settings)
    return _cached[key]


def get_summary_llm(
    settings: Annotated[Settings, Depends(get_settings)],
    planner: Annotated[LLMProvider, Depends(get_llm)],
) -> LLMProvider:
    """The model that writes summaries; the planning model unless a separate one is set."""
    if settings.summary_model_name == settings.model_name:
        return planner
    key = _cache_key(settings, settings.summary_model_name, True)
    if key not in _cached:
        _cached[key] = make_provider(settings, summary=True)
    return _cached[key]
