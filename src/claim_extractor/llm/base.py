"""Provider-agnostic interface to a chat LLM."""

from abc import ABC, abstractmethod
from typing import Any, Literal, TypedDict


class Message(TypedDict):
    """One chat message, in the format shared by Ollama and Mistral."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(Exception):
    """The LLM backend could not be reached or returned an unusable response."""


class LLMClient(ABC):
    """A chat model that answers with JSON text.

    Implementations must call the model with temperature 0 and constrain its
    output to `json_schema` when the backend supports it. The returned text is
    never trusted: the caller always validates it with Pydantic.
    """

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def complete(self, messages: list[Message], json_schema: dict[str, Any]) -> str:
        """Send the conversation and return the raw text of the model's reply.

        Raises:
            LLMError: on network, HTTP or response-format errors.
        """
