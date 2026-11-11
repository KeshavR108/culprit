"""Ollama client (local models, zero cost) using only the standard library."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from ..schemas import ToolCall
from .base import ChatResponse

HOST = "http://127.0.0.1:11434"


class OllamaLLM:
    def __init__(self, model: str = "qwen2.5-coder:3b", num_ctx: int = 16384, seed: int = 0,
                 temperature: float = 0.0, host: str = HOST, timeout: float = 600, num_predict: int = 1024):
        self.name = model
        self.model, self.num_ctx, self.seed, self.temperature = model, num_ctx, seed, temperature
        self.host, self.timeout, self.num_predict = host, timeout, num_predict

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        body = {"model": self.model, "messages": messages, "tools": tools, "stream": False,
                "options": {"num_ctx": self.num_ctx, "seed": self.seed, "temperature": self.temperature,
                            "num_predict": self.num_predict}}
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 500:
                raise
            detail = e.read().decode(errors="replace")[:300]
            return ChatResponse(content=f"[unparseable model output: {detail}]", seconds=time.time() - t0)
        msg = data.get("message", {})
        calls = [ToolCall(name=c["function"]["name"], arguments=_as_dict(c["function"].get("arguments")))
                 for c in msg.get("tool_calls") or []]
        content = msg.get("content") or ""
        if not calls:
            calls = parse_text_tool_calls(content, {t["function"]["name"] for t in tools})
        return ChatResponse(content=content, tool_calls=calls,
                            prompt_tokens=data.get("prompt_eval_count", 0),
                            completion_tokens=data.get("eval_count", 0), seconds=time.time() - t0)


def _as_dict(a) -> dict:
    if isinstance(a, dict):
        return a
    try:
        return json.loads(a) if a else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def json_objects(text: str) -> list[dict]:
    """Every top-level JSON object embedded in text (in code fences, tags, or bare)."""
    out, dec, i = [], json.JSONDecoder(), 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                out.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1
    return out


def parse_text_tool_calls(text: str, tool_names: set[str]) -> list[ToolCall]:
    calls = []
    for obj in json_objects(re.sub(r"</?tool_call>", "", text)):
        name = obj.get("name") or obj.get("tool")
        if name in tool_names:
            calls.append(ToolCall(name=name, arguments=_as_dict(obj.get("arguments", obj.get("parameters", {})))))
    return calls


def available_models(host: str = HOST) -> list[str]:
    with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
        return [m["name"] for m in json.loads(r.read())["models"]]
