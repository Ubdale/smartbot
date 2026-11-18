"""Pick the LLM provider from settings (LLM_PROVIDER / LLM_MODEL)."""

from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.llm.base import LLMProvider, LLMUnavailableError
from app.llm.ollama_provider import OllamaProvider


def make_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            settings.llm_model,
            settings.ollama_base_url,
            num_gpu=settings.ollama_num_gpu,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    # Gemini and Groq (free tiers) arrive in Milestone 5 (hosted mode).
    raise LLMUnavailableError(f"LLM_PROVIDER={settings.llm_provider!r} isn't available yet.")


_cached: dict[tuple[str, str, str, str], LLMProvider] = {}


def get_llm(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    key = (
        settings.llm_provider,
        settings.llm_model,
        settings.ollama_base_url,
        str(settings.ollama_num_gpu),
    )
    if key not in _cached:
        _cached[key] = make_provider(settings)
    return _cached[key]
