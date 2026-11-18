import json
from typing import Any

import httpx
import pytest

from app.llm.base import ChatMessage, LLMUnavailableError
from app.llm.ollama_provider import OllamaProvider
from app.llm.prompts import plan_json_schema


def provider(handler: Any) -> OllamaProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaProvider("qwen2.5:7b-instruct", "http://ollama:11434/", client=client)


def test_plan_sends_schema_and_parses_reply() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["url"] = str(request.url)
        reply = {"metric": "count", "chart": "none"}
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps(reply)}, "prompt_eval_count": 812,
                  "eval_count": 41},
        )  # fmt: skip

    attempt = provider(handler).plan("How many orders?", 'Table "t" (1 rows).', [])
    assert seen["url"] == "http://ollama:11434/api/chat"
    assert seen["format"] == plan_json_schema()
    assert seen["options"]["temperature"] == 0
    assert seen["stream"] is False
    assert seen["messages"][0]["role"] == "system"
    assert seen["messages"][-1]["content"].endswith("Question: How many orders?")
    assert attempt.plan is not None and attempt.plan.metric == "count"
    assert (attempt.call.prompt_tokens, attempt.call.completion_tokens) == (812, 41)


def test_invalid_json_becomes_feedback_not_crash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": '{"metric": "median"}'}})

    attempt = provider(handler).plan("q", "schema", [])
    assert attempt.plan is None
    assert attempt.error is not None and "metric" in attempt.error


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")), "ollama serve"),
        (lambda r: httpx.Response(404, json={"error": "model not found"}), "ollama pull"),
        (lambda r: httpx.Response(500), "error (500)"),
    ],
)
def test_errors_are_friendly(handler: Any, message: str) -> None:
    with pytest.raises(LLMUnavailableError, match=message.replace("(", r"\(").replace(")", r"\)")):
        provider(handler).complete([ChatMessage("user", "hi")])


def test_schema_has_no_refs() -> None:
    assert "$ref" not in json.dumps(plan_json_schema())


def test_num_gpu_option_and_oom_hint() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(500, json={"error": "llama-server: CUDA error: out of memory"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = OllamaProvider("m", "http://x", client=client, num_gpu=0)
    with pytest.raises(LLMUnavailableError, match="OLLAMA_NUM_GPU=0"):
        p.complete([ChatMessage("user", "hi")])
    assert seen["options"]["num_gpu"] == 0


def test_timeout_is_friendly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMUnavailableError, match="took too long"):
        provider(handler).complete([ChatMessage("user", "hi")])


def test_prompt_is_compact() -> None:
    from app.llm.prompts import plan_messages

    msgs = plan_messages("q", 'Table "t" (1 rows). Columns:\n- a (number)', [])
    assert len(msgs) == 2  # one system message with the examples, one question
    assert sum(len(m.content) for m in msgs) < 4500  # ~1k tokens: CPU inference cost
