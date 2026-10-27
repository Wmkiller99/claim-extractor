"""Test doubles shared by several test modules."""

from typing import Any

from claim_extractor.llm.base import LLMClient, Message


class FakeLLM(LLMClient):
    """Replays scripted replies and records every conversation it receives."""

    def __init__(self, replies: list[str | Exception]) -> None:
        super().__init__(model="fake-model")
        self.replies = replies
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], json_schema: dict[str, Any]) -> str:
        self.calls.append(list(messages))  # copy: what was sent must not change afterwards
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
