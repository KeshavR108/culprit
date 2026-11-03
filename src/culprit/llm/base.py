"""Provider-neutral LLM interface: chat(messages, tools) -> ChatResponse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..schemas import ToolCall


@dataclass
class ChatResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse: ...
