"""LLMClient backed by a local Ollama server (https://ollama.com)."""

from typing import Any

import httpx

from claim_extractor.llm.base import LLMClient, LLMError, Message

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient(LLMClient):
    """Calls Ollama's /api/chat endpoint with structured outputs."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(model)
        # `transport` lets tests replace the network with httpx.MockTransport.
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def complete(self, messages: list[Message], json_schema: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "format": json_schema,  # structured outputs: decoding is constrained to the schema
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            response = self._http.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status, body = exc.response.status_code, exc.response.text
            raise LLMError(f"Ollama returned HTTP {status}: {body}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Ollama at {self._http.base_url}: {exc}") from exc

        try:
            content: str = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response: {response.text[:200]}") from exc
        return content
