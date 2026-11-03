"""Tool: cProfile both sides and report which functions got hotter."""

from pydantic import BaseModel, Field

from .. import sandbox

NAME = "profile"
DESCRIPTION = ("Profile the benchmark on commit_a and commit_b with cProfile. Returns the top functions by "
               "cumulative time on commit_b and the functions whose time grew the most. Defaults: the "
               "parent of the bisected culprit vs the culprit (or good vs bad if not bisected yet).")

REPS = 3
TOP = 15


class Args(BaseModel):
    commit_a: str | None = Field(None, description="baseline commit hash")
    commit_b: str | None = Field(None, description="slow commit hash")


def _key(r):
    return f"{r['file']}:{r['func']}"


def run(s, a: Args) -> dict:
    if a.commit_a or a.commit_b:
        ca = s.resolve(a.commit_a) if a.commit_a else s.good
        cb = s.resolve(a.commit_b) if a.commit_b else s.bad
    elif s.bisect and s.bisect.get("status") == "found":
        cb = s.bisect["full_hash"]
        ca = sandbox.git(s.repo, "rev-parse", f"{cb}^").strip()
    else:
        ca, cb = s.good, s.bad
    rows_a = s.worker_for_commit(ca).profile(REPS)
    rows_b = s.worker_for_commit(cb).profile(REPS)
    by_a = {_key(r): r for r in rows_a}

    def ms(x):
        return round(1000 * x / REPS, 3)

    top = [{"function": f"{r['file']}:{r['line']}({r['func']})", "calls": r["ncalls"] // REPS,
            "cum_ms": ms(r["cumtime"]), "self_ms": ms(r["tottime"])}
           for r in rows_b if r["in_repo"]][:TOP]
    changes = []
    for r in rows_b:
        old = by_a.get(_key(r))
        d_self = r["tottime"] - (old["tottime"] if old else 0.0)
        changes.append({"function": f"{r['file']}:{r['line']}({r['func']})",
                        "self_ms_before": ms(old["tottime"]) if old else None, "self_ms_after": ms(r["tottime"]),
                        "cum_ms_before": ms(old["cumtime"]) if old else None, "cum_ms_after": ms(r["cumtime"]),
                        "calls_before": old["ncalls"] // REPS if old else 0, "calls_after": r["ncalls"] // REPS,
                        "_d": d_self})
    changes.sort(key=lambda c: -c["_d"])
    for c in changes:
        c.pop("_d")
    s.profile = {"commit_a": ca[:10], "commit_b": cb[:10], "top_on_b": top, "biggest_increases": changes[:8]}
    return s.profile
