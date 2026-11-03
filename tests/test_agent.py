"""Phase 3 tests: the agent loop and handoff rules, driven by a scripted fake LLM."""

import json

import pytest

from culprit.agent import MAX_STEPS, investigate, parse_finding
from culprit.llm.base import ChatResponse
from culprit.llm.fake import FakeLLM
from culprit.llm.ollama_client import parse_text_tool_calls
from culprit.schemas import ToolCall
from culprit.session import Config


def cfg_for(toy, good=0, bad=-1):
    return Config(repo=toy["repo"], good=toy["shas"][good], bad=toy["shas"][bad], bench="common",
                  bench_file=toy["bench_file"], test_args=["-q"])


def call(name, **args):
    return ChatResponse(tool_calls=[ToolCall(name=name, arguments=args)])


def last_tool_result(messages, name):
    for m in reversed(messages):
        if m["role"] == "tool" and m["name"] == name:
            return json.loads(m["content"])


def final(commit_from_bisect=True, confidence="high", commit=None):
    def respond(messages):
        c = commit or last_tool_result(messages, "bisect_commits")["first_bad_commit"]
        return ChatResponse(content=json.dumps({
            "culprit_commit": c, "file": "src/toy/core.py", "line": 3,
            "explanation": "list membership is O(n), set is O(1)", "confidence": confidence}))
    return respond


FIX = call("propose_patch", file="src/toy/core.py", old_code="    lookup = list(ys)", new_code="    lookup = set(ys)")


def test_happy_path_is_verified(toy_repo, tmp_path):
    llm = FakeLLM([call("bisect_commits"),
                   lambda m: call("show_diff", commit=last_tool_result(m, "bisect_commits")["first_bad_commit"]),
                   FIX, call("verify_patch"), final()])
    r = investigate(cfg_for(toy_repo), llm, tmp_path / "trace.jsonl")
    assert r.status == "verified", r.reasons
    assert toy_repo["culprit"].startswith(r.finding.culprit_commit)
    assert r.verification.significant_speedup and r.verification.tests_passed
    events = [json.loads(l)["kind"] for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert "tool" in events and events[-1] == "report"


def test_no_regression_skips_the_agent(toy_repo):
    llm = FakeLLM([])
    r = investigate(cfg_for(toy_repo, good=2, bad=4), llm)
    assert r.status == "no_regression" and llm.calls == 0


def test_wrong_commit_is_handed_off(toy_repo):
    llm = FakeLLM([call("bisect_commits"), FIX, call("verify_patch"),
                   final(commit=toy_repo["shas"][1])])
    r = investigate(cfg_for(toy_repo), llm)
    assert r.status == "handoff"
    assert any("does not match bisect" in x for x in r.reasons), (r.reasons, r.bisect)


def test_low_confidence_is_handed_off(toy_repo):
    llm = FakeLLM([call("bisect_commits"), FIX, call("verify_patch"), final(confidence="low")])
    r = investigate(cfg_for(toy_repo), llm)
    assert r.status == "handoff" and any("confidence is low" in x for x in r.reasons)


def test_budget_exhausted_is_handed_off(toy_repo):
    llm = FakeLLM([call("search_code", pattern="x")] * (MAX_STEPS + 5))
    r = investigate(cfg_for(toy_repo), llm)
    assert r.status == "handoff" and r.steps == MAX_STEPS
    assert any("step budget" in x for x in r.reasons)
    assert r.bisect["status"] == "found"


def test_invalid_final_answer_gets_one_correction(toy_repo):
    llm = FakeLLM([call("bisect_commits"), FIX, call("verify_patch"),
                   ChatResponse(content="The culprit is the list change."), final()])
    r = investigate(cfg_for(toy_repo), llm)
    assert r.status == "verified"
    assert "not a valid tool call" in llm.seen[-1][-1]["content"]


def test_hallucinated_arguments_return_errors(toy_repo):
    llm = FakeLLM([call("show_diff", commit="abc123nothere"), call("read_file", path="../../../etc/passwd"),
                   call("bisect_commits"), FIX, call("verify_patch"), final()])
    r = investigate(cfg_for(toy_repo), llm)
    assert r.status == "verified"
    tool_msgs = [m for m in llm.seen[2] if m["role"] == "tool"]
    assert all("error" in json.loads(m["content"]) for m in tool_msgs)


def test_parse_finding_and_text_tool_calls():
    f = parse_finding('Here you go:\n```json\n{"culprit_commit": "abc", "file": "a.py", "line": "12", '
                      '"explanation": "x", "confidence": "High"}\n```')
    assert f.line == 12 and f.confidence == "high"
    with pytest.raises(ValueError):
        parse_finding("no json here")
    calls = parse_text_tool_calls('<tool_call>{"name": "show_diff", "arguments": {"commit": "ab"}}</tool_call>',
                                  {"show_diff"})
    assert calls[0].name == "show_diff" and calls[0].arguments == {"commit": "ab"}


def test_v2_evidence_gate_rejects_premature_answer(toy_repo):
    from culprit import prompts
    prompts.set_variant("v2")
    try:
        llm = FakeLLM([call("bisect_commits"), final(),
                       lambda m: call("show_diff", commit=last_tool_result(m, "bisect_commits")["first_bad_commit"]),
                       FIX, call("verify_patch"), final()])
        r = investigate(cfg_for(toy_repo), llm)
    finally:
        prompts.set_variant("v1")
    assert r.status == "verified"
    assert "not looked at the code" in llm.seen[2][-1]["content"]
