"""Tool: binary search over the commit range for the first slow commit."""

import math

from pydantic import BaseModel

NAME = "bisect_commits"
DESCRIPTION = ("Find the first commit that made the benchmark slower, by binary search between the known "
               "good and bad commits. Each step is a statistical measurement. Returns the culprit commit "
               "and the evidence for every step. Call this before reading code.")


class Args(BaseModel):
    pass


def bisect(s) -> dict:
    commits = s.commits_in_range()
    if not commits:
        return {"status": "error", "error": "no commits between good and bad"}
    max_steps = math.ceil(math.log2(len(commits))) + 2
    full = s.initial if s.initial is not None else s.compare(s.good, s.bad)
    split = math.sqrt(max(full.ratio, 1.0))
    lo, hi = -1, len(commits) - 1
    path = []
    while hi - lo > 1:
        if len(path) >= max_steps:
            return {"status": "inconclusive", "reason": "step limit reached", "path": path}
        mid = (lo + hi) // 2
        m = s.compare(s.good, commits[mid])
        side = "slow" if m.ci_low > split else "fast" if m.ci_high < split else "unclear"
        path.append({"commit": commits[mid][:10], "side": side, "ratio": round(m.ratio, 3),
                     "ci95": [round(m.ci_low, 3), round(m.ci_high, 3)]})
        if side == "slow":
            hi = mid
        elif side == "fast":
            lo = mid
        else:
            return {"status": "inconclusive", "reason": f"measurement at {commits[mid][:10]} too noisy",
                    "path": path}
    first_bad = commits[hi]
    subject = s_subject(s, first_bad)
    return {"status": "found", "first_bad_commit": first_bad[:10], "full_hash": first_bad, "split": round(split, 3),
            "subject": subject, "commits_searched": len(commits), "steps": len(path), "path": path}


def s_subject(s, sha: str) -> str:
    from .. import sandbox
    return sandbox.git(s.repo, "log", "-1", "--format=%s", sha).strip()


def run(s, a: Args) -> dict:
    if s.bisect is None or s.bisect.get("status") != "found":
        s.bisect = bisect(s)
    return {k: v for k, v in s.bisect.items() if k != "full_hash"}
