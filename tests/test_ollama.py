import json
from collections.abc import Callable

import httpx
import pytest

from claim_extractor.llm.base import LLMError, Message
from claim_extractor.llm.ollama import OllamaClient

SCHEMA = {"type": "object"}
MESSAGES: list[Message] = [{"role": "user", "content": "Bonjour"}]


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaClient:
    """An OllamaClient whose HTTP calls go to `handler` instead of the network."""
    return OllamaClient(
        model="qwen2.5:7b",
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )


def test_sends_a_deterministic_json_request_and_returns_the_content() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "{}"}})

    reply = make_client(handler).complete(MESSAGES, SCHEMA)

    assert reply == "{}"
    assert requests[0].url == "http://ollama.test/api/chat"
    assert json.loads(requests[0].content) == {
        "model": "qwen2.5:7b",
        "messages": MESSAGES,
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0},
    }


def test_http_error_raises_llm_error_with_status_and_body() -> None:
    client = make_client(lambda request: httpx.Response(404, json={"error": "model not found"}))

    with pytest.raises(LLMError, match="HTTP 404.*model not found"):
        client.complete(MESSAGES, SCHEMA)


def test_unreachable_server_raises_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    with pytest.raises(LLMError, match="Cannot reach Ollama"):
        make_client(handler).complete(MESSAGES, SCHEMA)


def test_unexpected_response_body_raises_llm_error() -> None:
    client = make_client(lambda request: httpx.Response(200, json={"done": True}))

    with pytest.raises(LLMError, match="Unexpected Ollama response"):
        client.complete(MESSAGES, SCHEMA)
