"""Measure how big each planted regression really is, and check it keeps the tests passing."""

import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_history import CASES, load_meta
from regressions import REGRESSIONS

from culprit import sandbox
from culprit.session import Config, Session

OUT = HERE.parent / "results" / "regression_calibration.json"


def main():
    out = {}
    for reg in REGRESSIONS:
        case = CASES / f"{reg.id}-0"
        meta = load_meta(case)
        cfg = Config(repo=case / "repo", good=meta["good"], bad=meta["culprit"], bench=meta["bench"],
                     bench_file=case / "bench.py", rounds=30, max_rounds=60)
        with Session(cfg) as s:
            m = s.compare(s.good, s.bad)
            wt = s.wt.get(s.bad)
            t = sandbox.run([cfg.python, "-m", "pytest", "-p", "no:cacheprovider", *shlex.split(meta["test_args"])],
                            cwd=wt, timeout=300, write_root=s.wt.root, env={"PYTHONPATH": str(wt / "src")})
        out[reg.id] = {"category": reg.category, "bench": reg.bench, "expected": reg.expected,
                       "ratio": round(m.ratio, 4), "ci95": [round(m.ci_low, 4), round(m.ci_high, 4)],
                       "verdict": m.verdict, "median_good_ms": round(m.median_a_ms, 3),
                       "median_bad_ms": round(m.median_b_ms, 3), "rounds": m.n_rounds,
                       "tests_pass_with_regression": t.returncode == 0,
                       "tests_tail": (t.stdout + t.stderr).strip().splitlines()[-1:] if t.stdout or t.stderr else [],
                       "above_threshold": m.ci_low > 1.05}
        print(f"{reg.id:22s} {m.ratio:7.3f}x  CI [{m.ci_low:.3f}, {m.ci_high:.3f}]  {m.verdict:13s} "
              f"tests {'pass' if t.returncode == 0 else 'FAIL'}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"env": m.env.model_dump(), "regressions": out}, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
