"""Phase 5 plots: how agent accuracy, time and efficiency change with model size."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PARAMS_B = {"0.5b": 0.49, "1.5b": 1.54, "3b": 3.09, "7b": 7.62, "14b": 14.8}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = [("culprit_commit", "culprit commit", "#2a78d6", "o"),
          ("root_cause_line_strict", "root-cause line (±1)", "#eb6834", "s"),
          ("verified_fix", "verified fix", "#1baf7a", "D")]


def main(variant: str = None):
    m = json.loads((RESULTS / "metrics.json").read_text())
    runs = {}
    for name, v in m.items():
        g = re.match(r"agent-(v\d+)-([\d.]+b)$", name)
        if g:
            runs.setdefault(g.group(1), {})[g.group(2)] = v
    variant = variant or max(runs, key=lambda k: (len(runs[k]), k))
    sizes = sorted(runs[variant], key=lambda s: PARAMS_B[s])
    xs = [PARAMS_B[s] for s in sizes]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)
    summary = {"variant": variant, "sizes": {}}

    ax = axes[0]
    for key, label, color, marker in SERIES:
        ys, lo, hi = [], [], []
        for s in sizes:
            w = runs[variant][s][key]
            ys.append(100 * w["rate"])
            lo.append(100 * (w["rate"] - w["ci95"][0]))
            hi.append(100 * (w["ci95"][1] - w["rate"]))
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, markersize=8, linewidth=2,
                    capsize=3, elinewidth=1, label=label, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.set_title("Accuracy on 36 detectable regressions\n(95% Wilson CIs)", fontsize=11, loc="left")
    ax.set_ylabel("% of cases")
    ax.set_ylim(-3, 105)
    ax.legend(frameon=False, fontsize=9, loc="center left")

    ax = axes[1]
    t = [runs[variant][s]["cost"]["wall_time_s"] for s in sizes]
    llm = [runs[variant][s]["cost"]["llm_seconds"] or 0 for s in sizes]
    ax.plot(xs, t, color="#2a78d6", marker="o", markersize=8, linewidth=2, label="total wall time",
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot(xs, llm, color="#eb6834", marker="s", markersize=8, linewidth=2, label="of which LLM time",
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    for x, y in zip(xs, t):
        ax.annotate(f"{y:.0f}s", (x, y), xytext=(0, 8), textcoords="offset points", ha="center",
                    color=INK2, fontsize=9)
    ax.set_title("Time per investigation (mean)", fontsize=11, loc="left")
    ax.set_ylabel("seconds")
    ax.set_ylim(0, max(t) * 1.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    eff = []
    for s in sizes:
        r = runs[variant][s]
        minutes = r["cost"]["wall_time_s"] * r["cost"]["cases"] / 60
        eff.append(r["verified_fix"]["k"] / minutes if minutes else 0)
    ax.bar(xs, eff, width=[x * 0.35 for x in xs], color="#1baf7a", edgecolor=SURFACE, linewidth=2)
    for x, y in zip(xs, eff):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 4), textcoords="offset points", ha="center",
                    color=INK2, fontsize=9)
    ax.set_title("Verified fixes per minute of investigation", fontsize=11, loc="left")
    ax.set_ylabel("fixes / minute")
    ax.set_ylim(0, max(eff + [0.01]) * 1.3)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([s.upper() for s in sizes])
        ax.set_xlabel("model size (qwen2.5-coder, parameters, log scale)")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.minorticks_off()
    fig.suptitle(f"Culprit agent {variant} on PerfBench, by model size (Apple M4 laptop, CPU+GPU via Ollama)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "model_size.png", dpi=150, facecolor=SURFACE)
    for s, e in zip(sizes, eff):
        r = runs[variant][s]
        summary["sizes"][s] = {"params_b": PARAMS_B[s], "culprit_commit": r["culprit_commit"],
                               "root_cause_line_strict": r["root_cause_line_strict"],
                               "verified_fix": r["verified_fix"], "false_fix": r["false_fix"],
                               "wall_time_s": r["cost"]["wall_time_s"], "verified_per_minute": round(e, 3)}
    (RESULTS / "model_size.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {RESULTS / 'model_size.png'} ({variant}: {', '.join(sizes)})")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
