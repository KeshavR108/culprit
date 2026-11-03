"""Tool: apply the proposed patch to a throwaway copy of the bad commit and re-measure."""

from pydantic import BaseModel

from .. import measure, sandbox
from ..schemas import Verification

NAME = "verify_patch"
DESCRIPTION = ("Apply the last proposed patch to a temporary copy of the bad commit, measure it against the "
               "bad commit and the good commit, and run the tests. Reports whether the fix is verified.")


class Args(BaseModel):
    pass


def verify(s) -> Verification:
    if s.patch is None:
        return Verification(applied=False, error="no patch proposed yet; call propose_patch first")
    h = s.patch_hash(s.patch.diff)
    if s.verification is not None and s.verified_patch_hash == h:
        return s.verification
    copy = s.wt.fresh(s.bad, "patch")
    patch_file = s.wt.root / f"apply-{h}.diff"
    patch_file.write_text(s.patch.diff)
    r = sandbox.run(["git", "apply", "--recount", str(patch_file)], cwd=copy)
    if r.returncode != 0:
        return Verification(applied=False, error=r.stderr.strip()[:400])
    try:
        w = s.worker(f"patch:{h}", copy)
        speedup = s.compare(s.bad, "patched", worker_b=w, label_b="patched", cache_key_b=f"patch:{h}")
        vs_good = s.compare(s.good, "patched", worker_b=w, label_b="patched", cache_key_b=f"patch:{h}")
    except sandbox.SandboxError as e:
        v = Verification(applied=True, error=f"patched benchmark failed: {e}"[:600])
        s.verification, s.verified_patch_hash = v, h
        return v
    v = Verification(applied=True, speedup=speedup, vs_good=vs_good,
                     significant_speedup=measure.is_significant_speedup(speedup))
    if v.significant_speedup:
        pythonpath = str(copy / "src") if (copy / "src").exists() else str(copy)
        t = sandbox.run([s.cfg.python, "-m", "pytest", "-p", "no:cacheprovider", *s.cfg.test_args],
                        cwd=copy, timeout=s.cfg.test_timeout, write_root=s.wt.root,
                        env={"PYTHONPATH": pythonpath})
        v.tests_passed = t.returncode == 0
        v.tests_output = (t.stdout + t.stderr).strip()[-800:]
    v.verified = bool(v.significant_speedup and v.tests_passed)
    s.verification, s.verified_patch_hash = v, h
    s.trace.log("verification", verified=v.verified, tests_passed=v.tests_passed,
                speedup=speedup.summary(), vs_good=vs_good.summary())
    return v


def run(s, a: Args) -> dict:
    v = verify(s)
    if v.error:
        return {"verified": False, "error": v.error}
    out = {"verified": v.verified, "significant_speedup": v.significant_speedup, "tests_passed": v.tests_passed,
           "bad_vs_patched": v.speedup.summary(), "good_vs_patched": v.vs_good.summary()}
    if v.tests_passed is False:
        out["tests_output"] = v.tests_output[-500:]
    return out
