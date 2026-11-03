"""Phase 1 tests: the measurement engine says regression / no_regression / inconclusive correctly."""

import random
import time

import numpy as np
import pytest

from culprit.measure import analyze, bootstrap_ratio_ci, compare, decide, is_significant_speedup


def test_decide_rule():
    assert decide(1.10, 1.30, 0.001, 0.05) == "regression"
    assert decide(1.10, 1.30, 0.05, 0.05) == "inconclusive"
    assert decide(0.97, 1.03, 0.4, 0.05) == "no_regression"
    assert decide(1.01, 1.09, 0.001, 0.05) == "inconclusive"


def test_bootstrap_ci_contains_true_ratio():
    rng = np.random.default_rng(1)
    a = rng.lognormal(np.log(10e6), 0.05, 30)
    b = rng.lognormal(np.log(20e6), 0.05, 30)
    lo, hi = bootstrap_ratio_ci(a, b)
    assert lo < 2.0 < hi and hi - lo < 0.2


def test_analyze_detects_clear_regression_and_speedup():
    rng = np.random.default_rng(2)
    a = rng.normal(10e6, 0.2e6, 20)
    b = rng.normal(15e6, 0.2e6, 20)
    assert analyze(a, b).verdict == "regression"
    m = analyze(b, a)
    assert m.verdict == "no_regression" and is_significant_speedup(m)


def test_outlier_does_not_fake_a_regression():
    rng = np.random.default_rng(3)
    a = rng.normal(10e6, 0.1e6, 20)
    b = rng.normal(10e6, 0.1e6, 20)
    b[0] = 500e6
    assert analyze(a, b).verdict == "no_regression"


def _work():
    """~2 ms of real CPU work."""
    s = 0
    for i in range(60_000):
        s += i * i
    return s


def timed(fn):
    def run():
        t0 = time.perf_counter_ns()
        fn()
        return time.perf_counter_ns() - t0
    return run


@pytest.mark.slow
def test_identical_code_is_no_regression_at_least_95_percent():
    runs = 40
    verdicts = [compare(timed(_work), timed(_work), seed=i).verdict for i in range(runs)]
    no_reg = verdicts.count("no_regression")
    assert verdicts.count("regression") <= 2
    assert no_reg / runs >= 0.95, verdicts


@pytest.mark.slow
def test_injected_sleep_doubling_runtime_is_always_regression():
    t0 = time.perf_counter()
    _work()
    base = time.perf_counter() - t0

    def slower():
        _work()
        time.sleep(base)

    for i in range(20):
        assert compare(timed(_work), timed(slower), seed=i).verdict == "regression"


@pytest.mark.slow
def test_heavy_noise_is_inconclusive_not_a_false_alarm():
    rng = random.Random(0)

    def noisy():
        time.sleep(rng.uniform(0.001, 0.02))

    verdicts = [compare(timed(noisy), timed(noisy), seed=i).verdict for i in range(10)]
    assert "regression" not in verdicts
    assert verdicts.count("inconclusive") >= 8, verdicts
