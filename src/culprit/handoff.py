"""Deterministic rules that decide 'verified' vs 'hand off to a human'."""

from __future__ import annotations

from . import sandbox
from .schemas import Finding, Report
from .tools import bisect_commits, propose_patch, verify_patch


def decide(s, finding: Finding | None, budget_exhausted: bool) -> Report:
    r = Report(status="handoff", initial=s.initial, finding=finding)
    if s.initial is not None and s.initial.verdict == "no_regression":
        r.status, r.reasons = "no_regression", ["benchmark is not significantly slower (CI below threshold)"]
        return r
    if s.initial is None or s.initial.verdict == "inconclusive":
        r.reasons = ["measurements too noisy to confirm a regression"]
        return r

    reasons = []
    if budget_exhausted or finding is None:
        reasons.append("agent did not produce a final answer within the step budget")

    try:
        if s.bisect is None:
            s.bisect = bisect_commits.bisect(s)
    except Exception as e:
        s.bisect = {"status": "error", "error": str(e)}
    r.bisect = {k: v for k, v in s.bisect.items() if k != "full_hash"}
    r.profile = s.profile
    if s.bisect.get("status") != "found":
        reasons.append(f"bisect did not confirm a culprit ({s.bisect.get('status')})")

    if finding is not None:
        try:
            named = s.resolve(finding.culprit_commit or "")
        except ValueError:
            named = None
        if s.bisect.get("status") == "found" and named != s.bisect["full_hash"]:
            reasons.append("agent's culprit commit does not match bisect")
        if finding.confidence == "low":
            reasons.append("agent's confidence is low")
        if finding.patch and (s.patch is None or finding.patch.strip() != s.patch.diff.strip()):
            try:
                s.patch = propose_patch.validate(s, finding.patch)
            except (ValueError, sandbox.SandboxError) as e:
                reasons.append(f"patch in final answer is invalid: {e}"[:200])

    r.patch = s.patch
    if s.patch is None:
        reasons.append("no valid patch proposed")
    elif s.patch.lines_changed > propose_patch.MAX_LINES_CHANGED:
        reasons.append("patch too large")
    else:
        try:
            r.verification = verify_patch.verify(s)
        except Exception as e:
            reasons.append(f"verification failed to run: {e}"[:200])
        v = r.verification
        if v is not None:
            if v.error:
                reasons.append(f"patch could not be verified: {v.error}"[:200])
            elif not v.significant_speedup:
                reasons.append("patch gives no significant speedup")
            elif not v.tests_passed:
                reasons.append("patch breaks tests")

    r.reasons = reasons
    r.status = "verified" if not reasons else "handoff"
    return r
