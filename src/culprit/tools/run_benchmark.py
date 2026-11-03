"""Tool: measure one commit against another with the noise-aware measurement engine."""

from pydantic import BaseModel, Field

NAME = "run_benchmark"
DESCRIPTION = ("Measure the benchmark on commit_b against commit_a (baseline) with repeated interleaved "
               "runs. Returns the slowdown ratio, its 95% confidence interval and a verdict "
               "(regression / no_regression / inconclusive). Defaults: the known good and bad commits.")


class Args(BaseModel):
    commit_a: str | None = Field(None, description="baseline commit hash (default: known good commit)")
    commit_b: str | None = Field(None, description="commit to test (default: known bad commit)")


def run(s, a: Args) -> dict:
    ca = s.resolve(a.commit_a) if a.commit_a else s.good
    cb = s.resolve(a.commit_b) if a.commit_b else s.bad
    m = s.compare(ca, cb)
    return {"commit_a": ca[:10], "commit_b": cb[:10], "verdict": m.verdict,
            "ratio": round(m.ratio, 3), "ci95": [round(m.ci_low, 3), round(m.ci_high, 3)],
            "median_ms": [round(m.median_a_ms, 3), round(m.median_b_ms, 3)],
            "p_slower": float(f"{m.p_slower:.3g}"), "rounds": m.n_rounds}
