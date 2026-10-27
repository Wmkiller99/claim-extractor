import json
from collections.abc import Callable

import httpx
import pytest

from claim_extractor.llm.base import LLMError, Message
from claim_extractor.llm.mistral import MistralClient

API_KEY = "test-key"
SCHEMA = {"type": "object"}
MESSAGES: list[Message] = [{"role": "user", "content": "Bonjour"}]


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> MistralClient:
    """A MistralClient whose HTTP calls go to `handler` instead of the network."""
    return MistralClient(
        model="mistral-small-latest",
        api_key=API_KEY,
        base_url="https://mistral.test/v1",
        transport=httpx.MockTransport(handler),
    )


def reply_with(content: object) -> httpx.Response:
    message = {"role": "assistant", "content": content}
    return httpx.Response(200, json={"choices": [{"message": message}]})


def test_sends_an_authenticated_deterministic_json_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return reply_with("{}")

    reply = make_client(handler).complete(MESSAGES, SCHEMA)

    assert reply == "{}"
    assert requests[0].url == "https://mistral.test/v1/chat/completions"
    assert requests[0].headers["Authorization"] == f"Bearer {API_KEY}"
    assert json.loads(requests[0].content) == {
        "model": "mistral-small-latest",
        "messages": MESSAGES,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "claim_extraction", "schema": SCHEMA, "strict": True},
        },
    }


def test_http_error_raises_llm_error_without_leaking_the_key() -> None:
    client = make_client(lambda request: httpx.Response(401, json={"message": "Unauthorized"}))

    with pytest.raises(LLMError, match="HTTP 401.*Unauthorized") as exc_info:
        client.complete(MESSAGES, SCHEMA)

    assert API_KEY not in str(exc_info.value)


def test_unreachable_server_raises_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    with pytest.raises(LLMError, match="Cannot reach Mistral"):
        make_client(handler).complete(MESSAGES, SCHEMA)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, text="<html>Bad gateway</html>"),
        reply_with(None),
        reply_with([{"type": "text", "text": "{}"}]),
    ],
    ids=["no_choices", "not_json", "null_content", "chunk_list"],
)
def test_unexpected_response_raises_llm_error(response: httpx.Response) -> None:
    client = make_client(lambda request: response)

    with pytest.raises(LLMError, match="Unexpected Mistral response"):
        client.complete(MESSAGES, SCHEMA)
