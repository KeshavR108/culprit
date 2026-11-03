# Culprit

> **Note:** this README was generated with AI (Claude) from the project's code and the result files in `results/`. Every number below comes from `results/metrics.json`.

Culprit looks at a benchmark that got slower between two commits and tries to work out why. It checks that the slowdown is real, finds the commit that caused it, looks for the line, proposes a fix, and measures whether the fix actually helps. If it can't back up an answer with measurements, it hands the case to a person instead of guessing.

Everything runs locally. The LLM part uses small models through [Ollama](https://ollama.com) (`qwen2.5-coder`), so there are no API costs.

## How it works

```
good commit A, bad commit B, a benchmark
            │
            ▼
1. MEASURE (plain statistics, no LLM)
   A and B timed in alternating rounds → medians, bootstrap 95% CI for the ratio, Mann-Whitney test
   → regression / no regression / inconclusive
            │ regression
            ▼
2. INVESTIGATE (the LLM picks tools, up to 15 steps)
   bisect_commits · show_diff · read_file · search_code · profile · run_benchmark
   propose_patch · verify_patch
            │
            ▼
3. DECIDE (fixed rules, no LLM)
   "verified" only if bisect agrees with the LLM's commit, the patch is significantly
   faster than the bad commit, and the tests still pass. Anything else → hand off with evidence.
```

A few design choices:

- **Interleaved timing.** A and B alternate (A,B / B,A / ...), so laptop drift like heat or background apps hits both sides equally.
- **Three verdicts, not two.** "Inconclusive" is a real answer on a noisy laptop, and it leads to a hand-off rather than a guess.
- **Bisect splits at √R.** Each bisect step asks whether a commit looks like the good side or the bad side, using the full slowdown R. It doesn't use the 5% regression threshold, which made bisect give up on obvious cases.
- **The LLM never executes anything.** It asks for a tool, and the code validates the arguments and runs it in a sandbox: allowlisted commands only, temp git worktrees, timeouts, no network, no writes outside a temp dir (macOS `sandbox-exec`).
- **Nothing touches your repo** unless you pass `--apply` and type `apply`. Even then it doesn't commit.

More detail: [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Results

To check how often it's right, I built **PerfBench** from my [HNSW engine](https://github.com/keshavr108/semantic-paper-search). It has 63 git histories of 16-32 commits each:

- **51 histories have one planted slowdown** at a random position, hidden among harmless commits (comments, docs, tests, renames). About half the slow commits also contain an unrelated edit to the same file. Of the 51:
  - **36 are clearly detectable** (1.10× to 9.4× slower, measured in `results/regression_calibration.json`),
  - **15 are deliberately tiny** (0-6%), to test sensitivity.
- **12 control histories** have no slowdown at all.

All planted changes keep the test suite passing. Each case records the true commit and line, so grading is automatic.

Results on the 36 detectable regressions (95% Wilson intervals are in `results/metrics.json`):

| system | culprit commit | root-cause line (±5) | line (±1) | verified fix | false fix | avg time |
|---|---|---|---|---|---|---|
| classic: bisect + profiler + simple rule, no LLM | 100% (36/36) | 97% (35/36) | 97% (35/36) | no fixes | – | 15 s |
| LLM only, given the true diff (3B) | given | 100% (36/36) | 0% (0/36) | 8% (3/36) | 0/3 | 71 s |
| Culprit v1 (3B) | 100% (36/36) | 0% (0/36) | 0% (0/36) | 0% (0/36) | – | 4 s |
| Culprit v2 (3B) | 39% (14/36) | 36% (13/36) | 6% (2/36) | 3% (1/36) | 0/1 | 37 s |
| Culprit v3 (3B) | 67% (24/36) | 61% (22/36) | 11% (4/36) | 6% (2/36) | 0/2 | 70 s |

Measurement engine, the same for every system: **36/36** detectable regressions detected, **0/12** false alarms on the controls, and 3 of the 15 tiny probes flagged.

How to read this:

- **The deterministic tools do most of the locating.** Bisect found the right commit in every case, and once you have the commit, a simple "changed line inside the function that got hotter" rule finds the line 97% of the time. On this benchmark the LLM isn't needed to locate the problem. The open question is whether it can explain and fix it, and with a 3B model, mostly it can't yet.
- **The ±5-line rule is too generous.** The LLM-only baseline scores 100% at ±5 but 0% at ±1, because it cites the start of the diff hunk, three context lines above the change. That's why both columns are shown.
- **The agent's commit score is lower than bisect's** because the agent is scored on its *final answer*. When it runs out of steps without answering, that counts as wrong, even though bisect had found the commit.
- **No false fixes.** Across every run, nothing was marked "verified" when it was wrong. Verified fixes were also rare, so this says more about the hand-off rules being strict than about the agent being good.

### What changed between versions

Each change came from looking at why the previous version failed (`perfbench/failures.py` sorts every miss into a category from its trace):

- **v1 → v2, evidence gate.** In all 36 cases, v1 called bisect, then immediately answered without looking at any code, and made up a file name (`search.py:100`). v2 refuses a final answer until the agent has looked at the code, proposed a patch, and verified it.
- **v2 → v3, tolerant patch input.** In v2, 20 of 35 misses were the model looping on `propose_patch` because it pasted code with diff `+`/`-` markers or line-number prefixes. v3 cleans those up and finds the intended lines. It also fixes a bug where a copied line with lost indentation produced a broken patch without any error.

Where v3 still fails (34 misses):

| failure | count |
|---|---|
| patch didn't make it faster (often the model "fixed" the wrong line or re-wrote the slow code) | 10 |
| looped without progress | 10 |
| patch crashed when run | 8 |
| patch broke tests | 2 |
| right commit, wrong line | 2 |
| ran out of steps | 2 |

A model-size comparison (0.5B / 1.5B / 3B) is running and will be added here.

## Running it

Needs Python 3.11+, git, and Ollama with a model pulled (`ollama pull qwen2.5-coder:3b`).

```bash
make setup      # venv, pinned deps, builds PerfBench (clones the HNSW repo)
make test       # unit tests
make demo       # one planted regression in the HNSW engine, end to end
```

On your own repo:

```bash
.venv/bin/culprit investigate path/to/repo --bench search --bench-file bench.py --good <fast commit> --bad <slow commit>
```

`bench.py` defines `bench_<name>()` functions. Each one does its setup and returns a zero-argument function, and that function is what gets timed. Keep the bench file outside the repo's history, so every commit is measured the same way. See [TESTING.md](TESTING.md) for a walkthrough, including planting your own regression.

Reproducing the numbers:

```bash
python perfbench/run_suite.py --system agent --variant v3 --model qwen2.5-coder:3b --cases all --out results/runs/agent-v3-3b
python perfbench/analyze.py
```

## Limitations

- **Small, synthetic benchmark.** 36 detectable cases, all in two source files (~350 lines) of one project, so the confidence intervals are wide. The planted changes are realistic, but I wrote them.
- **Line-finding is easy here.** A culprit commit is usually small, which is why the no-LLM baseline does so well. Real regressions spread over large commits would be harder.
- **Python only, and laptop timing.** All numbers were measured on an Apple M4 laptop. A sub-5% slowdown is intentionally reported as "no regression".
- **Small models only so far** (up to 3B). Larger local models are the obvious next experiment.

## Layout

```
src/culprit/      measure.py (statistics) · agent.py (loop) · handoff.py (rules) · sandbox.py · tools/ · llm/
perfbench/        regressions.py · make_history.py · calibrate.py · run_suite.py · analyze.py · failures.py
results/          calibration, metrics, per-case results and step-by-step traces for every run
tests/            unit tests (fake LLM for the agent loop, real git repos for the tools)
```
