"""LLMClient backed by the Mistral API (https://docs.mistral.ai)."""

from typing import Any

import httpx

from claim_extractor.llm.base import LLMClient, LLMError, Message

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"


class MistralClient(LLMClient):
    """Calls Mistral's chat completions endpoint with structured outputs."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(model)
        # The key only lives in this header: it is never logged or put in an error message.
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def complete(self, messages: list[Message], json_schema: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "claim_extraction", "schema": json_schema, "strict": True},
            },
        }
        try:
            response = self._http.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status, body = exc.response.status_code, exc.response.text
            raise LLMError(f"Mistral returned HTTP {status}: {body}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Mistral at {self._http.base_url}: {exc}") from exc

        # `content` can also be a list of chunks or null in Mistral's API: only accept text.
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            content = None
        if not isinstance(content, str):
            raise LLMError(f"Unexpected Mistral response: {response.text[:200]}")
        return content
