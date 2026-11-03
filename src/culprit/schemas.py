"""Data shapes passed between the measurement engine, tools, agent, and report."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["regression", "no_regression", "inconclusive"]
Confidence = Literal["low", "medium", "high"]


class Environment(BaseModel):
    cpu: str
    cores: int
    python: str
    platform: str
    load_avg_1m: float


class Measurement(BaseModel):
    """Result of comparing benchmark timings of side A (baseline) vs side B."""

    verdict: Verdict
    ratio: float = Field(description="median(B) / median(A); 2.0 means B takes twice as long")
    ci_low: float = Field(description="lower end of the 95% bootstrap CI for the ratio")
    ci_high: float
    p_slower: float = Field(description="one-sided Mann-Whitney p-value for 'B slower than A'")
    p_faster: float = Field(description="one-sided Mann-Whitney p-value for 'B faster than A'")
    median_a_ms: float
    median_b_ms: float
    n_rounds: int
    tau: float
    env: Environment | None = None
    label_a: str = "A"
    label_b: str = "B"

    def summary(self) -> str:
        return (
            f"{self.label_a} {self.median_a_ms:.2f} ms -> {self.label_b} {self.median_b_ms:.2f} ms, "
            f"ratio {self.ratio:.3f} (95% CI {self.ci_low:.3f}-{self.ci_high:.3f}), "
            f"p_slower={self.p_slower:.2g}, n={self.n_rounds} rounds -> {self.verdict}"
        )


class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)


class Finding(BaseModel):
    """What the agent concludes."""

    culprit_commit: str | None = Field(None, description="hash of the commit that introduced the slowdown")
    file: str | None = Field(None, description="repo-relative path of the root-cause line")
    line: int | None = Field(None, description="line number of the root cause in that file")
    explanation: str = Field("", description="why this change makes the benchmark slower")
    patch: str | None = Field(None, description="unified diff that fixes it (optional if propose_patch was used)")
    confidence: Confidence = "low"


class Patch(BaseModel):
    diff: str
    files: list[str]
    lines_changed: int


class Verification(BaseModel):
    applied: bool
    tests_passed: bool | None = None
    tests_output: str = ""
    speedup: Measurement | None = None
    vs_good: Measurement | None = None
    significant_speedup: bool = False
    verified: bool = False
    error: str | None = None


class Report(BaseModel):
    status: Literal["verified", "handoff", "no_regression"]
    reasons: list[str] = Field(default_factory=list)
    initial: Measurement | None = None
    bisect: dict | None = None
    profile: dict | None = None
    finding: Finding | None = None
    patch: Patch | None = None
    verification: Verification | None = None
    steps: int = 0
    tool_calls: int = 0
    benchmark_comparisons: int = 0
    llm_tokens: dict = Field(default_factory=dict)
    wall_time_s: float = 0.0
    trace_path: str | None = None
