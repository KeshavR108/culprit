"""Investigation state shared by all tools: the repo, the commits, workers, budgets, results."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import measure, sandbox
from .schemas import Measurement, Patch, Verification
from .trace import Trace


@dataclass
class Config:
    repo: Path
    good: str
    bad: str
    bench: str
    bench_file: Path
    python: str = sys.executable
    test_args: list[str] = field(default_factory=lambda: ["-q", "-x"])
    tau: float = 0.05
    rounds: int = 15
    max_rounds: int = 40
    max_comparisons: int = 25
    test_timeout: float = 180
    cache_file: Path | None = None


class BudgetExceeded(Exception):
    pass


class Session:
    def __init__(self, cfg: Config, trace: Trace | None = None):
        self.cfg = cfg
        self.repo = Path(cfg.repo).resolve()
        self.trace = trace or Trace(None)
        self.wt = sandbox.Worktrees(self.repo)
        self.good = self.resolve(cfg.good)
        self.bad = self.resolve(cfg.bad)
        self._workers: dict[str, sandbox.BenchWorker] = {}
        self._cache = self._load_cache()
        self.comparisons = 0
        self.initial: Measurement | None = None
        self.bisect: dict | None = None
        self.profile: dict | None = None
        self.patch: Patch | None = None
        self.verification: Verification | None = None
        self.verified_patch_hash: str | None = None


    def resolve(self, ref: str) -> str:
        """Full hash for a commit name; raises ValueError with a helpful message."""
        ref = str(ref).strip()
        if not ref or ref.startswith("-"):
            raise ValueError(f"invalid commit {ref!r}")
        try:
            return sandbox.git(self.repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").strip()
        except sandbox.SandboxError:
            raise ValueError(f"unknown commit {ref!r}; use a hash from the commit list") from None

    def commits_in_range(self) -> list[str]:
        """Commits after good up to and including bad, oldest first."""
        out = sandbox.git(self.repo, "rev-list", "--first-parent", "--reverse", f"{self.good}..{self.bad}")
        return out.split()

    def commit_list_text(self, limit: int = 40) -> str:
        out = sandbox.git(self.repo, "log", "--first-parent", "--format=%h %s", f"{self.good}..{self.bad}")
        lines = out.strip().splitlines()
        return "\n".join(lines[:limit]) + (f"\n... ({len(lines) - limit} more)" if len(lines) > limit else "")


    def worker(self, key: str, checkout: Path) -> sandbox.BenchWorker:
        if key not in self._workers:
            w = sandbox.BenchWorker(checkout, Path(self.cfg.bench_file).resolve(), self.cfg.bench,
                                    self.cfg.python, self.wt.root)
            self.trace.log("worker_started", key=key, setup_s=w.info.get("setup_s"))
            self._workers[key] = w
        return self._workers[key]

    def worker_for_commit(self, sha: str) -> sandbox.BenchWorker:
        return self.worker(sha, self.wt.get(sha))

    def compare(self, a: str, b: str, worker_b: sandbox.BenchWorker | None = None,
                label_b: str | None = None, cache_key_b: str | None = None) -> Measurement:
        """Measure commit a (baseline) vs commit b (or a custom worker, e.g."""
        key = f"{self.cfg.bench}|{a}|{cache_key_b or b}|{self.cfg.tau}"
        if key in self._cache:
            m = Measurement.model_validate(self._cache[key])
            self.trace.log("measurement", cached=True, summary=m.summary())
            return m
        if self.comparisons >= self.cfg.max_comparisons:
            raise BudgetExceeded(f"benchmark budget of {self.cfg.max_comparisons} comparisons used up")
        self.comparisons += 1
        wa = self.worker_for_commit(a)
        wb = worker_b or self.worker_for_commit(b)
        m = measure.compare(wa.run_once, wb.run_once, rounds=self.cfg.rounds,
                            max_rounds=self.cfg.max_rounds, tau=self.cfg.tau,
                            label_a=a[:8], label_b=label_b or b[:8])
        self._cache[key] = m.model_dump()
        self._save_cache()
        self.trace.log("measurement", cached=False, summary=m.summary())
        return m

    def _load_cache(self) -> dict:
        f = self.cfg.cache_file
        if f and Path(f).exists():
            return json.loads(Path(f).read_text())
        return {}

    def _save_cache(self) -> None:
        if self.cfg.cache_file:
            Path(self.cfg.cache_file).write_text(json.dumps(self._cache, indent=1))


    @staticmethod
    def patch_hash(diff: str) -> str:
        return hashlib.sha256(diff.encode()).hexdigest()[:16]

    def close(self) -> None:
        for w in self._workers.values():
            w.close()
        self._workers.clear()
        self.wt.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
