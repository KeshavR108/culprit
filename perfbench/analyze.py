"""Compute PerfBench metrics from saved per-case results."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from make_history import CASES

RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
LINE_TOL = 5


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "rate": round(p, 3), "ci95": [round(max(0, c - h), 3), round(min(1, c + h), 3)]}


def load_index() -> dict:
    return {m["id"]: m for m in json.loads((CASES / "index.json").read_text())}


def load_calibration() -> dict:
    return json.loads((RESULTS / "regression_calibration.json").read_text())["regressions"]


def load_run(name: str) -> dict:
    d = RUNS / name
    return {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}


def commit_ok(res, meta) -> bool:
    c = ((res.get("finding") or {}).get("culprit_commit") or "").strip().lower()
    return len(c) >= 7 and meta["culprit"].startswith(c)


def norm_path(p: str | None) -> str:
    p = (p or "").strip().lstrip("./")
    return p


def line_exact(res, meta) -> bool:
    """Stricter line check: within +-1 of a planted line."""
    f = res.get("finding") or {}
    if norm_path(f.get("file")) != meta["planted_head"]["file"] or not isinstance(f.get("line"), int):
        return False
    lines = meta["planted_head"]["lines"] + meta["planted_culprit"]["lines"]
    return any(abs(f["line"] - ln) <= 1 for ln in lines)


def line_ok(res, meta) -> bool:
    f = res.get("finding") or {}
    if norm_path(f.get("file")) != meta["planted_head"]["file"] or not isinstance(f.get("line"), int):
        return False
    lines = meta["planted_head"]["lines"] + meta["planted_culprit"]["lines"]
    return any(abs(f["line"] - ln) <= LINE_TOL for ln in lines)


HIDDEN = '-q -k "ef200 or 5k or non_decreasing or better_connected"'


def run_hidden_tests(run_name: str) -> None:
    """Apply each verified patch to the bad commit and run the held-out slow tests (cached in the result)."""
    from culprit import sandbox
    from culprit.sandbox import Worktrees
    index = load_index()
    for p in sorted((RUNS / run_name).glob("*.json")):
        res = json.loads(p.read_text())
        if res.get("status") != "verified" or "hidden_tests_passed" in res:
            continue
        meta = index.get(res["case"])
        wt = Worktrees(CASES / res["case"] / "repo")
        try:
            copy = wt.fresh(meta["bad"], "hidden")
            pf = wt.root / "fix.diff"
            pf.write_text(res["patch"]["diff"])
            sandbox.run(["git", "apply", "--recount", str(pf)], cwd=copy)
            t = sandbox.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *shlex.split(HIDDEN)],
                            cwd=copy, timeout=900, write_root=wt.root, env={"PYTHONPATH": str(copy / "src")})
            res["hidden_tests_passed"] = t.returncode == 0
            res["hidden_tests_tail"] = (t.stdout + t.stderr).strip()[-300:]
        finally:
            wt.cleanup()
        p.write_text(json.dumps(res, indent=1, default=str))
        print(f"hidden tests {res['case']}: {'pass' if res['hidden_tests_passed'] else 'FAIL'}", flush=True)


def metrics(run: dict, index: dict, calib: dict) -> dict:
    det = [c for c, m in index.items() if m["kind"] == "regression" and calib[m["regression"]]["above_threshold"]]
    sub = [c for c, m in index.items() if m["kind"] == "regression" and not calib[m["regression"]]["above_threshold"]]
    ctl = [c for c, m in index.items() if m["kind"] == "control"]
    have = lambda ids: [c for c in ids if c in run]
    det, sub, ctl = have(det), have(sub), have(ctl)

    def initial_verdict(c):
        return (run[c].get("initial") or {}).get("verdict")

    out = {"n_cases": len(run), "n_detectable": len(det), "n_subthreshold": len(sub), "n_controls": len(ctl)}
    out["detection"] = wilson(sum(initial_verdict(c) == "regression" for c in det), len(det))
    out["false_alarm"] = wilson(sum(initial_verdict(c) == "regression" for c in ctl), len(ctl))
    out["subthreshold_flagged"] = wilson(sum(initial_verdict(c) == "regression" for c in sub), len(sub))
    out["errors"] = sum(run[c].get("status") == "error" for c in run)

    c_ok = {c: commit_ok(run[c], index[c]) for c in det}
    l_ok = {c: line_ok(run[c], index[c]) for c in det}
    verified = {c: run[c].get("status") == "verified" for c in run}
    oracle = any(r.get("system") == "notools" for r in run.values())
    out["culprit_commit"] = {"given": True} if oracle else wilson(sum(c_ok.values()), len(det))
    out["root_cause_line"] = wilson(sum(l_ok.values()), len(det))
    out["root_cause_line_strict"] = wilson(sum(line_exact(run[c], index[c]) for c in det), len(det))
    out["verified_fix"] = wilson(sum(verified[c] for c in det), len(det))

    false_fix = []
    for c in run:
        if not verified[c]:
            continue
        m = index[c]
        wrong = m["kind"] != "regression" or not commit_ok(run[c], m) or not line_ok(run[c], m) \
            or run[c].get("hidden_tests_passed") is False
        if wrong:
            false_fix.append(c)
    n_ver = sum(verified.values())
    out["false_fix"] = {"cases": false_fix, **(wilson(len(false_fix), n_ver) or {"k": 0, "n": 0, "rate": None})}
    out["hidden_tests"] = {"checked": sum("hidden_tests_passed" in run[c] for c in run),
                           "failed": [c for c in run if run[c].get("hidden_tests_passed") is False]}

    ho = [c for c in det if run[c].get("status") == "handoff"]
    out["handoffs"] = {"total": len(ho), "necessary": sum(not (c_ok[c] and l_ok[c]) for c in ho),
                       "unnecessary_correct_diagnosis": sum(c_ok[c] and l_ok[c] for c in ho)}

    cal = {}
    for conf in ("low", "medium", "high"):
        ids = [c for c in det if (run[c].get("finding") or {}).get("confidence") == conf]
        cal[conf] = wilson(sum(c_ok[c] and l_ok[c] for c in ids), len(ids))
    out["calibration"] = cal

    def mean(key, ids, f=lambda r, k: r.get(k)):
        vals = [f(run[c], key) for c in ids if f(run[c], key) is not None]
        return round(statistics.mean(vals), 2) if vals else None
    inv = [c for c in det if run[c].get("status") != "no_regression"]
    out["cost"] = {"cases": len(inv),
                   "steps": mean("steps", inv), "tool_calls": mean("tool_calls", inv),
                   "benchmark_comparisons_measured": mean("benchmark_comparisons", inv),
                   "wall_time_s": mean("run_seconds", inv),
                   "llm_seconds": mean("llm_seconds", inv, lambda r, k: (r.get("llm_tokens") or {}).get(k)),
                   "prompt_tokens": mean("prompt", inv, lambda r, k: (r.get("llm_tokens") or {}).get(k)),
                   "completion_tokens": mean("completion", inv, lambda r, k: (r.get("llm_tokens") or {}).get(k))}
    out["by_category"] = {}
    for c in det:
        cat = index[c]["category"]
        b = out["by_category"].setdefault(cat, {"n": 0, "commit": 0, "line": 0, "verified": 0})
        b["n"] += 1
        b["commit"] += c_ok[c]
        b["line"] += l_ok[c]
        b["verified"] += verified[c]
    return out


def pct(w) -> str:
    if w and w.get("given"):
        return "given (oracle)"
    if not w or w.get("rate") is None:
        return "n/a"
    return f"{100 * w['rate']:.0f}% ({w['k']}/{w['n']})"


def table(all_metrics: dict) -> str:
    rows = ["| system | detection | false alarms | culprit commit | root-cause line (±5) | line (±1) | verified fix | false fix | avg time / investigation |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, m in all_metrics.items():
        rows.append(f"| {name} | {pct(m['detection'])} | {pct(m['false_alarm'])} | {pct(m['culprit_commit'])} | "
                    f"{pct(m['root_cause_line'])} | {pct(m['root_cause_line_strict'])} | {pct(m['verified_fix'])} | {pct(m['false_fix'])} | "
                    f"{m['cost']['wall_time_s']} s |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", action="store_true", help="run held-out tests on verified patches first")
    args = p.parse_args()
    index, calib = load_index(), load_calibration()
    runs = sorted(d.name for d in RUNS.iterdir() if d.is_dir() and any(d.glob("*.json")))
    if args.hidden:
        for r in runs:
            run_hidden_tests(r)
    all_m = {r: metrics(load_run(r), index, calib) for r in runs}
    (RESULTS / "metrics.json").write_text(json.dumps(all_m, indent=2))
    md = ["# PerfBench results (generated by perfbench/analyze.py; do not edit by hand)", "",
          f"Cases: {next(iter(all_m.values()))['n_detectable'] if all_m else 0} detectable regressions, "
          f"sub-threshold probes and controls as listed in metrics.json. Rates with 95% Wilson CIs in metrics.json.",
          "", table(all_m), ""]
    (RESULTS / "results_table.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
