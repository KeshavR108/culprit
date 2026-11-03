"""How often does the measurement engine cry wolf on IDENTICAL code?"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_history import CASES, load_meta

from culprit import measure
from culprit.session import Config, Session

TRIALS = 30


def main():
    case = CASES / "control-00"
    meta = load_meta(case)
    out = {}
    for bench in ("search", "build"):
        cfg = Config(repo=case / "repo", good=meta["good"], bad=meta["good"], bench=bench,
                     bench_file=case / "bench.py")
        verdicts, ratios = [], []
        with Session(cfg) as s:
            wa = s.worker("a", s.wt.fresh(s.good, "a"))
            wb = s.worker("b", s.wt.fresh(s.good, "b"))
            for i in range(TRIALS):
                m = measure.compare(wa.run_once, wb.run_once, seed=i)
                verdicts.append(m.verdict)
                ratios.append(round(m.ratio, 4))
                print(bench, i, m.summary(), flush=True)
        out[bench] = {"trials": TRIALS, "regression": verdicts.count("regression"),
                      "no_regression": verdicts.count("no_regression"),
                      "inconclusive": verdicts.count("inconclusive"), "ratios": ratios,
                      "false_alarm_rate": verdicts.count("regression") / TRIALS, "env": m.env.model_dump()}
    total = sum(v["regression"] for v in out.values())
    out["overall"] = {"trials": 2 * TRIALS, "false_alarms": total, "rate": total / (2 * TRIALS)}
    (HERE.parent / "results" / "false_alarm.json").write_text(json.dumps(out, indent=2))
    print(out["overall"])


if __name__ == "__main__":
    main()
