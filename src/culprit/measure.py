"""Measurement engine: is B really slower than A, or is it noise?"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Callable

import numpy as np
from scipy.stats import mannwhitneyu

from .schemas import Environment, Measurement

Runner = Callable[[], int]

N_BOOT = 2000
ALPHA_P = 0.01


def bootstrap_ratio_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = 0):
    """95% CI for median(b)/median(a) by resampling both sides independently."""
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(a), size=(n_boot, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_boot, len(b)))
    ratios = np.median(b[idx_b], axis=1) / np.median(a[idx_a], axis=1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(lo), float(hi)


def decide(ci_low: float, ci_high: float, p_slower: float, tau: float) -> str:
    """Verdict: regression if CI_low > 1+tau and p < 0.01; no_regression if CI_high < 1+tau; else inconclusive."""
    if ci_low > 1 + tau and p_slower < ALPHA_P:
        return "regression"
    if ci_high < 1 + tau:
        return "no_regression"
    return "inconclusive"


def analyze(t_a, t_b, tau: float = 0.05, seed: int = 0, **labels) -> Measurement:
    """Turn two lists of timings (ns) into a Measurement with a verdict."""
    a = np.asarray(t_a, dtype=float)
    b = np.asarray(t_b, dtype=float)
    ci_low, ci_high = bootstrap_ratio_ci(a, b, seed=seed)
    p_slower = float(mannwhitneyu(b, a, alternative="greater").pvalue)
    p_faster = float(mannwhitneyu(b, a, alternative="less").pvalue)
    return Measurement(
        verdict=decide(ci_low, ci_high, p_slower, tau),
        ratio=float(np.median(b) / np.median(a)),
        ci_low=ci_low,
        ci_high=ci_high,
        p_slower=p_slower,
        p_faster=p_faster,
        median_a_ms=float(np.median(a)) / 1e6,
        median_b_ms=float(np.median(b)) / 1e6,
        n_rounds=len(a),
        tau=tau,
        **labels,
    )


def is_significant_speedup(m: Measurement, tau: float | None = None) -> bool:
    """Mirror image of 'regression': B is confidently faster than A by more than tau."""
    tau = m.tau if tau is None else tau
    return m.ci_high < 1 / (1 + tau) and m.p_faster < ALPHA_P


def compare(
    run_a: Runner,
    run_b: Runner,
    rounds: int = 15,
    max_rounds: int = 40,
    warmup: int = 3,
    tau: float = 0.05,
    step: int = 5,
    seed: int = 0,
    label_a: str = "A",
    label_b: str = "B",
) -> Measurement:
    """Measure A vs B with interleaved runs; add rounds while the answer is unclear."""
    for _ in range(warmup):
        run_a()
        run_b()

    t_a: list[int] = []
    t_b: list[int] = []

    def do_rounds(k: int) -> None:
        for _ in range(k):
            if len(t_a) % 2 == 0:
                t_a.append(run_a())
                t_b.append(run_b())
            else:
                t_b.append(run_b())
                t_a.append(run_a())

    do_rounds(rounds)
    m = analyze(t_a, t_b, tau, seed, label_a=label_a, label_b=label_b)
    while m.verdict == "inconclusive" and len(t_a) < max_rounds:
        do_rounds(min(step, max_rounds - len(t_a)))
        m = analyze(t_a, t_b, tau, seed, label_a=label_a, label_b=label_b)
    m.env = environment()
    return m


def environment() -> Environment:
    """Record where a measurement was taken, so numbers are never quoted without context."""
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout.strip() or cpu
        elif os.path.exists("/proc/cpuinfo"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return Environment(
        cpu=cpu,
        cores=os.cpu_count() or 1,
        python=platform.python_version(),
        platform=platform.platform(),
        load_avg_1m=os.getloadavg()[0] if hasattr(os, "getloadavg") else -1.0,
    )
