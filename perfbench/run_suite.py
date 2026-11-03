"""Run a system on PerfBench cases and save one JSON result per case (resumable)."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_history import CASES, load_meta

from culprit import agent, prompts, sandbox
from culprit.llm.ollama_client import OllamaLLM, json_objects
from culprit.schemas import Finding
from culprit.session import Config, Session
from culprit.tools import bisect_commits, profile, propose_patch, verify_patch
from culprit.trace import Trace

SMOKE = ["debug_log-1", "heuristic_recompute-0", "control-00", "prepare_in_loop-0", "ef_default-0"]


def case_config(case: Path, meta: dict) -> Config:
    return Config(repo=case / "repo", good=meta["good"], bad=meta["bad"], bench=meta["bench"],
                  bench_file=case / "bench.py", test_args=shlex.split(meta["test_args"]),
                  cache_file=case / "measure_cache.json")


def run_classic(case: Path, meta: dict, trace_path: Path) -> dict:
    t0 = time.time()
    with Session(case_config(case, meta), Trace(trace_path)) as s:
        s.initial = s.compare(s.good, s.bad)
        out = {"status": "no_regression" if s.initial.verdict == "no_regression" else "handoff",
               "initial": s.initial.model_dump(exclude={"env"}), "finding": None,
               "reasons": ["classic tooling gives no explanation or fix"]}
        if s.initial.verdict == "regression":
            s.bisect = bisect_commits.bisect(s)
            out["bisect"] = {k: v for k, v in s.bisect.items() if k != "full_hash"}
            if s.bisect["status"] == "found":
                prof = profile.run(s, profile.Args())
                file, line = heuristic_line(s, s.bisect["full_hash"], prof)
                out["finding"] = Finding(culprit_commit=s.bisect["full_hash"], file=file, line=line,
                                         confidence="medium").model_dump()
                out["profile"] = prof
        out["benchmark_comparisons"] = s.comparisons
    out["wall_time_s"] = round(time.time() - t0, 2)
    return out


def heuristic_line(s, sha: str, prof: dict):
    """Changed line (culprit-commit coordinates) inside the function whose self time grew most."""
    diff = sandbox.git(s.repo, "show", "--format=", "--unified=0", sha)
    changed, file, lineno = [], None, 0
    for l in diff.splitlines():
        if l.startswith("+++ "):
            file = l[6:] if l.startswith("+++ b/") else None
        elif l.startswith("@@"):
            lineno = int(re.search(r"\+(\d+)", l).group(1))
        elif file and file.endswith(".py") and not file.startswith("tests/"):
            if l.startswith("+"):
                changed.append((file, lineno))
                lineno += 1
            elif l.startswith("-") and not l.startswith("---"):
                changed.append((file, lineno))
    if not changed:
        return None, None
    for inc in prof.get("biggest_increases", []):
        m = re.match(r"(.+):(\d+)\((.+)\)$", inc["function"])
        if not m:
            continue
        f, def_line = m.group(1), int(m.group(2))
        cands = [ln for cf, ln in changed if cf == f and ln >= def_line]
        if cands:
            return f, min(cands)
    return changed[0]


NOTOOLS_PROMPT = """This commit made a benchmark slower. Explain why and fix it.

{diff}

Reply with ONLY this JSON object:
{{"file": "<path>", "line": <line number in the new version of the file>, "explanation": "<why it is slower>",
 "old_code": "<exact current lines to replace>", "new_code": "<replacement lines>", "confidence": "low|medium|high"}}"""


def run_notools(case: Path, meta: dict, model: str, trace_path: Path) -> dict:
    t0 = time.time()
    trace = Trace(trace_path)
    with Session(case_config(case, meta), trace) as s:
        s.initial = s.compare(s.good, s.bad)
        out = {"status": "no_regression" if s.initial.verdict == "no_regression" else "handoff",
               "initial": s.initial.model_dump(exclude={"env"}), "finding": None, "reasons": []}
        if s.initial.verdict != "regression" or meta["kind"] != "regression":
            out["reasons"].append("not run: no confirmed regression with a known culprit")
            out["wall_time_s"] = round(time.time() - t0, 2)
            return out
        diff = sandbox.git(s.repo, "show", "--format=commit %H%nSubject: %s%n", meta["culprit"])[:8000]
        resp = OllamaLLM(model).chat([{"role": "user", "content": NOTOOLS_PROMPT.format(diff=diff)}], [])
        trace.log("llm", content=resp.content)
        out["llm_tokens"] = {"prompt": resp.prompt_tokens, "completion": resp.completion_tokens,
                             "llm_seconds": resp.seconds}
        objs = [o for o in json_objects(resp.content) if "file" in o or "explanation" in o]
        if not objs:
            out["reasons"].append("no parsable JSON answer")
        else:
            o = objs[-1]
            try:
                f = Finding(culprit_commit=meta["culprit"], file=o.get("file"), line=o.get("line"),
                            explanation=str(o.get("explanation", "")),
                            confidence=str(o.get("confidence", "low")).lower().strip())
            except Exception as e:
                f = Finding(culprit_commit=meta["culprit"], explanation=str(o.get("explanation", "")))
                out["reasons"].append(f"invalid fields: {e}"[:200])
            out["finding"] = f.model_dump()
            try:
                d = propose_patch.edit_to_diff(s.wt.get(s.bad), str(o.get("file") or ""),
                                               str(o.get("old_code") or ""), str(o.get("new_code") or ""))
                s.patch = propose_patch.validate(s, d)
                v = verify_patch.verify(s)
                out["patch"] = s.patch.model_dump()
                out["verification"] = v.model_dump()
                if v.verified and f.confidence != "low":
                    out["status"] = "verified"
                else:
                    out["reasons"].append("fix not verified" if not v.verified else "confidence low")
            except Exception as e:
                out["reasons"].append(f"patch invalid: {e}"[:200])
        out["benchmark_comparisons"] = s.comparisons
    out["wall_time_s"] = round(time.time() - t0, 2)
    return out


def run_agent(case: Path, meta: dict, model: str, trace_path: Path, variant: str) -> dict:
    prompts.set_variant(variant)
    r = agent.investigate(case_config(case, meta), OllamaLLM(model), trace_path)
    return json.loads(r.model_dump_json())


def select_cases(spec: str) -> list[str]:
    if spec == "smoke":
        return SMOKE
    if spec == "dev":
        return sorted(p.name for p in CASES.glob("dev-*"))
    index = [m for m in json.loads((CASES / "index.json").read_text())]
    if spec == "all":
        return [m["id"] for m in index]
    if spec == "regressions":
        return [m["id"] for m in index if m["kind"] == "regression"]
    return spec.split(",")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=["classic", "notools", "agent"], required=True)
    p.add_argument("--model", default="qwen2.5-coder:3b")
    p.add_argument("--variant", default="v1")
    p.add_argument("--cases", default="smoke")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    propose_patch.TOLERANT = args.variant not in ("v1", "v2")
    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    ids = select_cases(args.cases)
    for i, cid in enumerate(ids, 1):
        dest = out / f"{cid}.json"
        if dest.exists():
            continue
        case = CASES / cid
        meta = load_meta(case)
        trace_path = out / "traces" / f"{cid}.jsonl"
        t0 = time.time()
        try:
            if args.system == "classic":
                res = run_classic(case, meta, trace_path)
            elif args.system == "notools":
                res = run_notools(case, meta, args.model, trace_path)
            else:
                res = run_agent(case, meta, args.model, trace_path, args.variant)
        except Exception:
            res = {"status": "error", "error": traceback.format_exc()[-3000:]}
        res.update({"case": cid, "system": args.system, "model": None if args.system == "classic" else args.model,
                    "variant": args.variant if args.system == "agent" else None,
                    "run_seconds": round(time.time() - t0, 2)})
        dest.write_text(json.dumps(res, indent=1, default=str))
        f = res.get("finding") or {}
        print(f"[{i}/{len(ids)}] {cid:24s} {res['status']:13s} commit={str(f.get('culprit_commit'))[:8]} "
              f"line={f.get('file')}:{f.get('line')} {res['run_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
