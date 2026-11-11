"""Collect resume-ready facts from saved results only (results/resume_facts.json)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def pct(w):
    return None if not w or w.get("rate") is None else round(100 * w["rate"], 1)


def main():
    m = json.loads((R / "metrics.json").read_text())
    fa = json.loads((R / "false_alarm.json").read_text())
    cal = json.loads((R / "regression_calibration.json").read_text())["regressions"]
    index = json.loads((ROOT / "perfbench" / "cases" / "index.json").read_text())
    best = m["agent-v3-3b"]
    facts = {
        "benchmark": {
            "cases_total": len(index),
            "regression_cases": sum(c["kind"] == "regression" for c in index),
            "detectable_regressions": best["n_detectable"],
            "subthreshold_probes": best["n_subthreshold"],
            "control_cases": best["n_controls"],
            "regression_types": len(cal),
            "slowdown_range_detectable": [min(v["ratio"] for v in cal.values() if v["above_threshold"]),
                                          max(v["ratio"] for v in cal.values() if v["above_threshold"])],
            "target": "keshavr108/semantic-paper-search (HNSW engine) @ 5cc47e0",
        },
        "measurement_engine": {
            "detection_pct": pct(best["detection"]), "detection": best["detection"],
            "false_alarms_on_controls": best["false_alarm"],
            "false_alarms_identical_code": fa["overall"],
        },
        "systems": {name: {"culprit_commit_pct": None if (v["culprit_commit"] or {}).get("given") else pct(v["culprit_commit"]),
                           "root_cause_line_pm5_pct": pct(v["root_cause_line"]),
                           "root_cause_line_pm1_pct": pct(v["root_cause_line_strict"]),
                           "verified_fix": v["verified_fix"], "false_fix": {k: v["false_fix"].get(k) for k in ("k", "n")},
                           "avg_seconds": v["cost"]["wall_time_s"]}
                    for name, v in m.items()},
        "improvement_v1_to_v3_3b": {
            "root_cause_line_pm5_pct": [pct(m["agent-v1-3b"]["root_cause_line"]), pct(best["root_cause_line"])],
            "verified_fix_k": [m["agent-v1-3b"]["verified_fix"]["k"], best["verified_fix"]["k"]],
        },
        "false_fixes_all_runs": sum(v["false_fix"].get("k") or 0 for v in m.values()),
        "verified_claims_all_runs": sum(v["false_fix"].get("n") or 0 for v in m.values()),
        "real_users_or_repos": 0,
    }
    (R / "resume_facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2)[:3000])


if __name__ == "__main__":
    main()
