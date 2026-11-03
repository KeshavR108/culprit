"""Classify every miss of an agent run into one failure category, from its trace."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import RESULTS, RUNS, commit_ok, line_ok, load_calibration, load_index

ORDER = ["not detected", "bisect inconclusive / wrong (noise)", "looped without progress",
         "answered before looking at code", "budget exhausted", "wrong commit", "right commit, wrong line",
         "could not write an applicable patch", "patch failed to run", "patch didn't help", "patch broke tests",
         "low confidence (correct diagnosis)", "false fix: verified but wrong"]


def trace_events(run: str, case: str) -> list[dict]:
    p = RUNS / run / "traces" / f"{case}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def classify(res: dict, meta: dict, events: list[dict]) -> tuple[str, str]:
    tools = [e for e in events if e["kind"] == "tool"]
    names = [t["name"] for t in tools]
    finding = res.get("finding") or {}
    reasons = res.get("reasons", [])
    v = res.get("verification") or {}
    if (res.get("initial") or {}).get("verdict") != "regression":
        return "not detected", f"initial verdict {(res.get('initial') or {}).get('verdict')}"
    b = res.get("bisect") or {}
    if b.get("status") != "found" or not meta["culprit"].startswith(str(b.get("first_bad_commit", "-"))):
        return "bisect inconclusive / wrong (noise)", f"bisect {b.get('status')} {b.get('first_bad_commit')}"
    if res.get("status") == "verified":
        if commit_ok(res, meta) and line_ok(res, meta) and res.get("hidden_tests_passed") is not False:
            return "", ""
        return "false fix: verified but wrong", f"{finding.get('file')}:{finding.get('line')}"
    calls = [(t["name"], json.dumps(t["arguments"], sort_keys=True)) for t in tools]
    top = Counter(calls).most_common(1)
    if not finding and top and top[0][1] >= 4:
        return "looped without progress", f"{top[0][0][0]} called {top[0][1]}x with identical arguments"
    if finding and not set(names) & {"show_diff", "read_file", "search_code"}:
        return "answered before looking at code", f"tools {names}; claimed {finding.get('file')}:{finding.get('line')}"
    if not finding:
        return "budget exhausted", f"{len(names)} tool calls: {', '.join(names[-5:])}"
    if not commit_ok(res, meta):
        return "wrong commit", f"named {finding.get('culprit_commit')}"
    if not line_ok(res, meta):
        return "right commit, wrong line", (f"claimed {finding.get('file')}:{finding.get('line')}, planted "
                                            f"{meta['planted_head']['file']}:{meta['planted_head']['lines']}")
    if not res.get("patch"):
        errs = [t["result"][:120] for t in tools if t["name"] == "propose_patch"]
        return "could not write an applicable patch", errs[-1] if errs else "never called propose_patch"
    if v.get("error"):
        return "patch failed to run", v["error"][:160]
    if not v.get("significant_speedup"):
        return "patch didn't help", ((v.get("speedup") or {}).get("ratio") and
                                     f"bad->patched ratio {v['speedup']['ratio']:.3f}") or "no speedup"
    if v.get("tests_passed") is False:
        return "patch broke tests", (v.get("tests_output") or "")[-160:]
    if any("confidence is low" in r for r in reasons):
        return "low confidence (correct diagnosis)", ""
    return "other", "; ".join(reasons)[:160]


def analyze_run(run: str) -> dict:
    index, calib = load_index(), load_calibration()
    out = {"run": run, "cases": {}, "counts": Counter(), "hallucinated_args_cases": 0}
    for p in sorted((RUNS / run).glob("*.json")):
        res = json.loads(p.read_text())
        meta = index[res["case"]]
        ev = trace_events(run, res["case"])
        if any('"error"' in e.get("result", "") and ("unknown commit" in e["result"] or "no such file" in e["result"]
                                                      or "escapes" in e["result"]) for e in ev if e["kind"] == "tool"):
            out["hallucinated_args_cases"] += 1
        detectable = meta["kind"] == "regression" and calib[meta["regression"]]["above_threshold"]
        if not detectable and res.get("status") != "verified":
            continue
        cat, evidence = classify(res, meta, ev)
        if cat:
            out["cases"][res["case"]] = {"category": cat, "evidence": evidence}
            out["counts"][cat] += 1
    out["counts"] = {k: out["counts"][k] for k in ORDER + ["other"] if out["counts"].get(k)}
    (RESULTS / "failures").mkdir(exist_ok=True)
    (RESULTS / "failures" / f"{run}.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    for run in sys.argv[1:]:
        r = analyze_run(run)
        print(f"\n{run}: {sum(r['counts'].values())} misses; {r['hallucinated_args_cases']} cases with a hallucinated argument")
        for k, n in r["counts"].items():
            ex = next(c for c, v in r["cases"].items() if v["category"] == k)
            print(f"  {n:3d}  {k:40s} e.g. {ex}: {r['cases'][ex]['evidence'][:100]}")


if __name__ == "__main__":
    main()
