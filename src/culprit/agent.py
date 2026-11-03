"""The agent loop: a local LLM decides which tool to call next, step by step."""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import ValidationError

from . import handoff, prompts, tools
from .llm.base import LLM
from .llm.ollama_client import json_objects
from .schemas import Finding, Report
from .session import Config, Session
from .trace import Trace

MAX_STEPS = 15
MAX_TOOL_RESULT_CHARS = 7000


def parse_finding(text: str) -> Finding:
    """The final answer must be a JSON object matching the Finding schema."""
    for obj in reversed(json_objects(text)):
        if "culprit_commit" in obj or "explanation" in obj:
            if isinstance(obj.get("confidence"), str):
                obj["confidence"] = obj["confidence"].strip().lower()
            return Finding.model_validate(obj)
    raise ValueError("no JSON object with the Finding fields found")


def missing_evidence(s, used: set[str]) -> str | None:
    """v2 evidence gate: a final answer is only accepted once the evidence for it exists."""
    if s.bisect is None:
        return "You have not located the commit yet. Call bisect_commits first."
    if not used & {"show_diff", "read_file"}:
        commit = s.bisect.get("first_bad_commit", "")
        return (f"You have not looked at the code yet. Call show_diff with commit \"{commit}\" and find the "
                "changed line that makes it slower. Use the exact file path shown in the diff.")
    if s.patch is None:
        return ("You have not proposed a fix yet. Call propose_patch with file, old_code (lines copied exactly "
                "from show_diff or read_file, without the leading +/- or line numbers) and new_code.")
    if s.verification is None:
        return "You have not measured your fix yet. Call verify_patch."
    return None


def investigate(cfg: Config, llm: LLM, trace_path: Path | None = None, echo=None) -> Report:
    t0 = time.time()
    trace = Trace(trace_path, echo)
    with Session(cfg, trace) as s:
        s.initial = s.compare(s.good, s.bad)
        trace.log("initial_measurement", **s.initial.model_dump(exclude={"env"}))
        if s.initial.verdict != "regression":
            report = handoff.decide(s, None, budget_exhausted=False)
            return _finish(report, s, t0, trace, steps=0, calls=0, tokens={})

        messages = [{"role": "system", "content": prompts.SYSTEM},
                    {"role": "user", "content": prompts.task(cfg, s.initial.summary(), s.commit_list_text())}]
        schemas = tools.schemas()
        finding, n_calls, step = None, 0, 0
        used: set[str] = set()
        tokens = {"prompt": 0, "completion": 0, "llm_seconds": 0.0}
        for step in range(1, MAX_STEPS + 1):
            resp = llm.chat(messages, schemas)
            tokens["prompt"] += resp.prompt_tokens
            tokens["completion"] += resp.completion_tokens
            tokens["llm_seconds"] += resp.seconds
            trace.log("llm", step=step, content=resp.content[:2000],
                      tool_calls=[c.model_dump() for c in resp.tool_calls])

            if resp.tool_calls:
                messages.append({"role": "assistant", "content": resp.content,
                                 "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}}
                                                for c in resp.tool_calls]})
                for c in resp.tool_calls:
                    n_calls += 1
                    used.add(c.name)
                    result = tools.call(s, c.name, c.arguments)
                    text = json.dumps(result, default=str)[:MAX_TOOL_RESULT_CHARS]
                    trace.log("tool", step=step, name=c.name, arguments=c.arguments, result=text)
                    messages.append({"role": "tool", "name": c.name, "content": text})
                continue

            gate = missing_evidence(s, used) if prompts.VARIANT != "v1" else None
            if gate:
                trace.log("evidence_gate", step=step, message=gate)
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": gate})
                continue
            try:
                finding = parse_finding(resp.content)
                break
            except (ValueError, ValidationError) as e:
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": prompts.FIX_JSON + str(e)[:300]})
        else:
            step = MAX_STEPS

        trace.log("finding", finding=finding.model_dump() if finding else None)
        report = handoff.decide(s, finding, budget_exhausted=finding is None)
        return _finish(report, s, t0, trace, step, n_calls, tokens)


def _finish(report: Report, s: Session, t0: float, trace: Trace, steps, calls, tokens) -> Report:
    report.steps, report.tool_calls, report.llm_tokens = steps, calls, tokens
    report.benchmark_comparisons = s.comparisons
    report.wall_time_s = round(time.time() - t0, 2)
    report.trace_path = str(trace.path) if trace.path else None
    trace.log("report", status=report.status, reasons=report.reasons)
    return report
