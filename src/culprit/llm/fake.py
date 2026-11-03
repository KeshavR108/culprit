"""Scripted LLM for tests: replays a fixed list of responses, so the agent loop can be tested without a model."""

from __future__ import annotations

from .base import ChatResponse


class FakeLLM:
    name = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools):
        self.seen.append(list(messages))
        if self.calls >= len(self.script):
            return ChatResponse(content="I am not sure.")
        item = self.script[self.calls]
        self.calls += 1
        return item(messages) if callable(item) else item
