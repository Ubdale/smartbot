"""Gemini and Groq providers (mocked HTTP), and switching providers via settings only."""

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.llm.base import ChatMessage, LLMProvider, LLMUnavailableError
from app.llm.factory import make_provider
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.prompts import plan_json_schema

PLAN = {"metric": "count", "chart": "none"}
MESSAGES = [
    ChatMessage("system", "rules"),
    ChatMessage("user", "Question: a"),
    ChatMessage("assistant", "{}"),
    ChatMessage("user", "Question: b"),
]


def gemini_ok(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1381, "candidatesTokenCount": 51},
        },
    )


def gemini(
    handler: Any, model: str = "gemini-2.5-flash", sleeps: list[float] | None = None
) -> GeminiProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    record = sleeps.append if sleeps is not None else (lambda _s: None)
    return GeminiProvider(model, "test-key", client, sleep=record)


# --- Gemini ---------------------------------------------------------------------------


def test_gemini_request_shape_and_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["key"] = str(req.url), req.headers["x-goog-api-key"]
        seen["body"] = json.loads(req.content)
        return gemini_ok(json.dumps(PLAN))

    attempt = gemini(handler).plan("How many?", 'Table "t"', [])
    body = seen["body"]
    assert seen["url"].endswith("/models/gemini-2.5-flash:generateContent")
    assert seen["key"] == "test-key"
    assert body["systemInstruction"]["parts"][0]["text"].startswith("You turn a business")
    assert [c["role"] for c in body["contents"]] == ["user"]
    cfg = body["generationConfig"]
    assert cfg["responseMimeType"] == "application/json"
    assert cfg["responseJsonSchema"] == plan_json_schema()
    assert cfg["temperature"] == 0
    assert cfg["thinkingConfig"] == {"thinkingBudget": 0}  # 2.5 models: no thinking
    assert attempt.plan is not None
    assert attempt.plan.metric == "count"
    assert (attempt.call.prompt_tokens, attempt.call.completion_tokens) == (1381, 51)
    assert attempt.call.provider == "gemini"


def test_gemini_maps_roles_and_merges_system() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return gemini_ok("hi")

    gemini(handler, model="gemini-3.5-flash").complete(MESSAGES)
    assert [c["role"] for c in seen["contents"]] == ["user", "model", "user"]
    assert "thinkingConfig" not in seen["generationConfig"]
    assert "responseJsonSchema" not in seen["generationConfig"]


def test_gemini_retries_once_on_short_rate_limit() -> None:
    calls = []
    sleeps: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429,
                json={"error": {"details": [{"@type": "RetryInfo", "retryDelay": "3s"}]}},
            )
        return gemini_ok("ok")

    assert gemini(handler, sleeps=sleeps).complete(MESSAGES).text == "ok"
    assert sleeps == [3.0]


def test_gemini_long_rate_limit_fails_fast() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"details": [{"retryDelay": "40s"}]}})

    with pytest.raises(LLMUnavailableError, match="quota"):
        gemini(handler).complete(MESSAGES)


def test_gemini_overloaded_retries_then_explains() -> None:
    sleeps: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(LLMUnavailableError, match="overloaded"):
        gemini(handler, sleeps=sleeps).complete(MESSAGES)
    assert sleeps == [2.0]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(403), "API key"),
        (httpx.Response(404), "no model"),
        (httpx.Response(500), r"error \(500\)"),
        (httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}), "SAFETY"),
    ],
)
def test_gemini_errors(response: httpx.Response, message: str) -> None:
    with pytest.raises(LLMUnavailableError, match=message):
        gemini(lambda req: response).complete(MESSAGES)


# --- Groq -----------------------------------------------------------------------------


def test_groq_json_mode_with_schema_hint() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(PLAN)}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 30},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    attempt = GroqProvider("llama-3.3-70b-versatile", "gsk-test", client).plan("q", "schema", [])
    assert seen["auth"] == "Bearer gsk-test"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert '"metric"' in seen["body"]["messages"][0]["content"]  # schema appended to system
    assert attempt.plan is not None
    assert (attempt.call.prompt_tokens, attempt.call.provider) == (900, "groq")


@pytest.mark.parametrize(("status", "message"), [(429, "quota"), (401, "API key")])
def test_groq_errors(status: int, message: str) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(status)))
    with pytest.raises(LLMUnavailableError, match=message):
        GroqProvider("m", "k", client).complete(MESSAGES)


# --- switching providers needs no code changes -------------------------------------


@pytest.mark.parametrize(
    ("env", "cls", "model"),
    [
        ({}, OllamaProvider, "qwen2.5:7b-instruct"),
        ({"llm_provider": "gemini", "gemini_api_key": "k"}, GeminiProvider, "gemini-2.5-flash"),
        ({"llm_provider": "groq", "groq_api_key": "k"}, GroqProvider, "openai/gpt-oss-120b"),
        ({"llm_provider": "gemini", "gemini_api_key": "k", "llm_model": "gemini-3.5-flash"},
         GeminiProvider, "gemini-3.5-flash"),
    ],
)  # fmt: skip
def test_provider_selected_by_settings(env: dict[str, Any], cls: type, model: str) -> None:
    provider = make_provider(Settings(_env_file=None, **env))
    assert isinstance(provider, cls)
    assert provider.model == model


def test_missing_key_is_a_clear_error_in_dev() -> None:
    with pytest.raises(LLMUnavailableError, match="GEMINI_API_KEY"):
        make_provider(Settings(_env_file=None, llm_provider="gemini"))


def test_prod_refuses_to_start_without_provider_key() -> None:
    with pytest.raises(ValueError, match="needs its API key"):
        Settings(_env_file=None, app_env="prod", secret_key=SecretStr("x" * 40),
                 llm_provider="gemini")  # fmt: skip


def test_leftover_model_from_another_provider_is_caught() -> None:
    s = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="k",
                 llm_model="qwen2.5:7b-instruct")  # fmt: skip
    with pytest.raises(LLMUnavailableError, match="Leave LLM_MODEL empty"):
        make_provider(s)


def test_empty_llm_model_means_provider_default() -> None:
    s = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="k", llm_model="")
    assert s.model_name == "gemini-2.5-flash"


@pytest.mark.parametrize(
    ("env", "plan_model", "summary_model"),
    [
        ({"llm_provider": "gemini", "gemini_api_key": "k"}, "gemini-2.5-flash",
         "gemini-3.5-flash-lite"),  # separate free-tier quota for summaries
        ({"llm_provider": "gemini", "gemini_api_key": "k", "llm_model": "gemini-3.5-flash"},
         "gemini-3.5-flash", "gemini-3.5-flash"),  # custom model: used for both
        ({"llm_provider": "gemini", "gemini_api_key": "k", "llm_summary_model": "gemini-x"},
         "gemini-2.5-flash", "gemini-x"),
        ({}, "qwen2.5:7b-instruct", "qwen2.5:7b-instruct"),
    ],
)  # fmt: skip
def test_summary_model_selection(env: dict[str, Any], plan_model: str, summary_model: str) -> None:
    s = Settings(_env_file=None, **env)
    assert (s.model_name, s.summary_model_name) == (plan_model, summary_model)


def test_gemini_summarizer_never_waits_on_rate_limit() -> None:
    s = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="k")
    planner, summarizer = make_provider(s), make_provider(s, summary=True)
    assert isinstance(planner, GeminiProvider)
    assert isinstance(summarizer, GeminiProvider)
    assert (planner.max_retry_wait, summarizer.max_retry_wait) == (20, 0)


# --- fallback: gemini out of quota -> groq answers ------------------------------------


class _Down(LLMProvider):
    name, model = "down", "x"

    def complete(self, messages: list[ChatMessage], **kw: Any) -> Any:
        raise LLMUnavailableError("quota")


class _Up(LLMProvider):
    name, model = "up", "y"

    def complete(self, messages: list[ChatMessage], **kw: Any) -> Any:
        from app.llm.base import Completion

        return Completion(json.dumps(PLAN) if kw.get("json_schema") else "ok", 10, 2, 5)


def test_fallback_answers_and_logs_the_provider_that_answered() -> None:
    from app.llm.factory import FallbackProvider

    chain = FallbackProvider([_Down(), _Up()])
    attempt = chain.plan("q", "schema", [])
    assert attempt.plan is not None
    assert attempt.call.provider == "up"  # usage is attributed to the real answerer
    text, call = chain.summarize("q", "d", "t")
    assert (text, call.provider) == ("ok", "up")


def test_fallback_raises_last_error_when_all_fail() -> None:
    from app.llm.factory import FallbackProvider

    with pytest.raises(LLMUnavailableError, match="quota"):
        FallbackProvider([_Down(), _Down()]).complete(MESSAGES)


def test_fallback_settings_build_a_chain_where_gemini_does_not_wait() -> None:
    from app.llm.factory import FallbackProvider

    s = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="k",
                 llm_fallback_provider="groq", groq_api_key="g")  # fmt: skip
    planner = make_provider(s)
    assert isinstance(planner, FallbackProvider)
    gemini_p, groq_p = planner.providers
    assert isinstance(gemini_p, GeminiProvider)
    assert gemini_p.max_retry_wait == 0  # switch to the backup instead of waiting
    assert isinstance(groq_p, GroqProvider)
    assert groq_p.model == "openai/gpt-oss-120b"
    summarizer = make_provider(s, summary=True)
    assert isinstance(summarizer, FallbackProvider)
    assert [p.model for p in summarizer.providers] == [
        "gemini-3.5-flash-lite",
        "openai/gpt-oss-20b",
    ]


def test_prod_needs_fallback_key_too() -> None:
    with pytest.raises(ValueError, match="LLM_FALLBACK_PROVIDER=groq needs its API key"):
        Settings(
            _env_file=None,
            app_env="prod",
            secret_key=SecretStr("x" * 40),
            llm_provider="gemini",
            gemini_api_key="k",
            llm_fallback_provider="groq",
        )
