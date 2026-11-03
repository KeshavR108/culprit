"""Render a Report as Markdown (saved next to the trace) and for the terminal."""

from __future__ import annotations

from .schemas import Report

ICON = {"verified": "VERIFIED", "handoff": "HANDED OFF TO A HUMAN", "no_regression": "NO REGRESSION"}


def to_markdown(r: Report) -> str:
    out = [f"# Culprit report: {ICON[r.status]}", ""]
    if r.reasons:
        out += ["**Why not verified:**", *[f"- {x}" for x in r.reasons], ""]
    if r.initial:
        out += ["## 1. Is it really slower? (measurement engine)", "", f"`{r.initial.summary()}`", ""]
    if r.bisect:
        out += ["## 2. Which commit? (bisect)", ""]
        if r.bisect.get("status") == "found":
            out += [f"First bad commit **{r.bisect['first_bad_commit']}**: {r.bisect.get('subject', '')} "
                    f"({r.bisect['commits_searched']} commits searched in {r.bisect['steps']} measured steps)", ""]
        else:
            out += [f"Bisect status: {r.bisect.get('status')}: {r.bisect.get('reason', r.bisect.get('error', ''))}", ""]
        out += [f"Each step: is the commit's CI above or below the split point sqrt(full slowdown) = "
                f"{r.bisect.get('split', '?')}?", "",
                "| step | commit | ratio vs good | 95% CI | side |", "|---|---|---|---|---|"]
        for i, p in enumerate(r.bisect.get("path", []), 1):
            out.append(f"| {i} | {p['commit']} | {p['ratio']:.3f} | {p['ci95'][0]:.3f}-{p['ci95'][1]:.3f} | {p['side']} |")
        out.append("")
    if r.profile:
        out += ["## 3. Where does the time go? (profile, per benchmark run)", "",
                "| function | self ms before | self ms after | calls before | calls after |", "|---|---|---|---|---|"]
        for c in r.profile.get("biggest_increases", [])[:6]:
            out.append(f"| `{c['function']}` | {c['self_ms_before']} | {c['self_ms_after']} | "
                       f"{c['calls_before']} | {c['calls_after']} |")
        out.append("")
    if r.finding:
        f = r.finding
        out += ["## 4. Diagnosis (LLM)", "", f"- Culprit commit: `{f.culprit_commit}`",
                f"- Root cause: `{f.file}:{f.line}`", f"- Confidence: {f.confidence}", "", f"> {f.explanation}", ""]
    if r.patch:
        out += ["## 5. Proposed fix (not applied)", "", "```diff", r.patch.diff.rstrip(), "```", ""]
    if r.verification and r.verification.speedup:
        v = r.verification
        out += ["## 6. Verification (patched copy, measured)", "",
                f"- bad vs patched: `{v.speedup.summary()}`",
                f"- good vs patched: `{v.vs_good.summary()}`",
                f"- significant speedup: {v.significant_speedup}; tests passed: {v.tests_passed}", ""]
    out += ["## Cost", "", f"{r.steps} LLM steps, {r.tool_calls} tool calls, {r.benchmark_comparisons} "
            f"measured comparisons, {r.llm_tokens.get('prompt', 0)} prompt + "
            f"{r.llm_tokens.get('completion', 0)} completion tokens, {r.wall_time_s:.0f} s total.", ""]
    if r.trace_path:
        out.append(f"Full step trace: `{r.trace_path}`")
    return "\n".join(out)
