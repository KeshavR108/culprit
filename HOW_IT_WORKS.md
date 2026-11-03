# Culprit: what it does

Culprit answers one question: **"my benchmark got slower between commit A and commit B. Why?"**

You give it a git repo, a benchmark, a known-good commit and a known-bad commit. It:

1. proves the slowdown is real (statistics),
2. finds the commit that caused it (bisect),
3. finds the line and explains why it's slower (a local LLM),
4. proposes a fix and proves the fix works (measurement + tests),
5. or, if it can't be sure, **hands off to a human** with all the evidence instead of guessing.

It runs entirely on your laptop. The LLM is a small local model through Ollama (free, no API keys).

---

## Running it

```bash
.venv/bin/culprit investigate <repo> --bench search --bench-file <bench.py> --good <A> --bad <B>
```

| option | meaning |
|---|---|
| `<repo>` | path to a git repository |
| `--bench search` | which benchmark to run: the function `bench_search()` in the bench file |
| `--bench-file` | a Python file with `bench_<name>()` functions (see below) |
| `--good`, `--bad` | the fast commit and the slow commit |
| `--model` | Ollama model, default `qwen2.5-coder:3b` |
| `--test-args` | arguments for `pytest`, used to check a fix doesn't break anything |
| `--apply` | after a **verified** fix, offer to apply it (you must type `apply`; it never commits) |

Other commands:
- `culprit measure ...`: only step 1 (is B really slower than A?).
- `culprit demo`: runs everything on a planted regression in the HNSW engine.

### The benchmark file

```python
def bench_search():
    index = hnsw.load("index.npz")              # setup: runs once, NOT timed
    Q = np.load("queries.npy")
    return lambda: index.batch_search(Q, k=10)  # this function is what gets timed
```

The bench file lives **outside** the repo history, so every commit is measured with exactly
the same benchmark.

---

## What happens, step by step

### Step 1: Is it really slower? (deterministic, no LLM), `measure.py`

- Checks out A and B into temporary folders (`git worktree`); your working copy is never touched.
- Starts one long-lived worker process per commit, which runs the setup once.
- Warms up each side 3 times and throws those runs away.
- Times them **interleaved**: A,B then B,A then A,B... (15 rounds, up to 40 if the answer is unclear),
  so laptop drift (heat, background apps) hits both sides equally.
- Computes:
  - **ratio** = median(B) / median(A) (2.0 means twice as slow),
  - a **95% confidence interval** for the ratio (bootstrap, 2,000 resamples),
  - a **Mann-Whitney p-value** for "B is slower than A".
- Verdict (threshold τ = 5%):
  - `regression` if the CI's lower end is above 1.05 and p < 0.01,
  - `no_regression` if the CI's upper end is below 1.05,
  - `inconclusive` otherwise (too noisy to tell).

If the verdict is `no_regression`, Culprit stops and says so. If it's `inconclusive`, it hands off.
Only a confirmed `regression` goes on to the agent.

### Step 2: The agent loop (the LLM decides), `agent.py`

The LLM gets the task, the measurement and the commit list, then picks tools one at a time
(up to 15 steps). It can't run anything itself: it asks for a tool by name, and Culprit's code
validates the arguments, runs the tool in the sandbox and sends back the result.

| tool | what it does |
|---|---|
| `bisect_commits` | binary search over the commits for the first slow one. Each step measures a commit and asks "is its whole CI above or below the split point $\sqrt{\text{full slowdown}}$?". If a step can't tell, bisect reports `inconclusive` instead of guessing. |
| `show_diff` | the diff of one commit (size-limited) |
| `read_file` | lines of a file at a commit, with line numbers (max 150 lines) |
| `search_code` | regex search over the repo's files |
| `profile` | cProfile on both sides: top functions and which ones got hotter |
| `run_benchmark` | measure any two commits with the step-1 engine |
| `propose_patch` | the fix, as `file` + `old_code` + `new_code` (or a diff). Validated with `git apply --check`, **not applied**. Max 60 changed lines. |
| `verify_patch` | applies the fix to a throwaway copy of the bad commit, measures it against the bad and the good commit, and runs the tests |

The model finishes by replying with a JSON answer: culprit commit, file, line, explanation, confidence.

**Evidence gate (agent v2 and later):** the final answer is refused until the agent has bisected,
looked at the code, proposed a patch and verified it. Each refusal tells the model the next step
and uses up one of its 15 steps.

### Step 3: Verified or hand off (deterministic, no LLM), `handoff.py`

**Verified** only if ALL of these are true:
- the slowdown was confirmed in step 1,
- bisect found a culprit commit **and** the LLM named that same commit,
- the LLM's confidence isn't `low`,
- there is a valid patch of 60 changed lines or fewer,
- the patched copy is **significantly faster** than the bad commit (same statistics as step 1),
- the tests **pass** on the patched copy.

Otherwise: **hand off to a human**, with the exact reasons listed. If the LLM skipped bisect,
`handoff.py` runs it anyway so the human still gets that evidence.

### Output

Saved in `runs/<time>/`:
- `report.md`: measurement, bisect path (every step with its CI), profile changes, the LLM's
  diagnosis, the proposed patch, verification numbers, and cost (steps, tokens, time).
- `report.json`: the same, machine-readable.
- `trace.jsonl`: every step: what the LLM asked for, what each tool returned, timings.

---

## Safety rules (enforced in code)

- **Read-only by default.** It never changes your repo, commits, rewrites history or deletes files.
  Fixes are only applied with `--apply` plus typing `apply`.
- **Allowlisted commands only**: read-only `git` subcommands, `git worktree`/`git apply` inside its own
  temp folders, and Python running either `pytest` or Culprit's benchmark worker. No shell,
  no `python -c`, no other programs.
- **No network** for anything it runs (macOS `sandbox-exec`, plus a Python socket block).
- **No file writes outside its temp folder** for anything it runs.
- **Timeouts** on every command.
- **Paths are checked**: `../` escapes, absolute paths and symlinks pointing outside the repo are rejected.

---

## Who does what

| part | done by | why |
|---|---|---|
| is it slower? | statistics code | has an exact, checkable answer |
| which commit? | bisect code | has an exact, checkable answer |
| which line, why, what fix? | the LLM | needs reading and reasoning |
| is the fix real? | measurement + tests | can't be trusted to the LLM |
| verified or hand off? | fixed rules | a confident wrong answer is the worst outcome |

---

## PerfBench: how we measure how often it's right

`perfbench/` builds test cases from the HNSW engine
([semantic-paper-search](https://github.com/keshavr108/semantic-paper-search) @ 5cc47e0):

- **63 git histories** of 16-32 commits each: harmless commits (comments, docs, new tests, renames)
  with **one planted slowdown** hidden at a random position, plus **12 control** histories with
  no slowdown. About half the slow commits also contain an unrelated edit, so finding the commit
  doesn't automatically give you the line.
- **17 kinds of slowdown** (set → list, heap → sort, NumPy → Python loop, extra copies,
  logging in a hot loop, recomputing a cached value, a changed default...). Each one keeps all
  tests passing; it only makes the code slower.
- `calibrate.py` measures each slowdown's real size: 12 are clearly detectable (1.10× to 9.4×),
  5 are tiny (0-6%) and are used to test sensitivity.
- `run_suite.py` runs a system on every case (resumable); `analyze.py` computes every metric;
  `failures.py` puts each miss into a failure category from its trace.

Systems compared:
- **classic**: no LLM. Measurement, bisect and profiler, with a simple rule to pick the line. No fix.
- **notools**: the LLM sees only the true culprit diff and must explain and fix it.
- **agent v1 / v2 / v3**: Culprit. v2 adds the evidence gate; v3 makes `propose_patch` tolerant of
  the copy mistakes small models make (diff `+`/`-` markers, line-number prefixes, wrong indentation).

Metrics: detection rate, false alarms, culprit commit, root-cause line (±5 and strict ±1),
verified-fix rate, **false-fix rate** (claimed verified but wrong; must be ~0), handoff quality,
calibration by confidence, time and tokens. All rates come with 95% confidence intervals.

---

## Where things are

| file | what's in it |
|---|---|
| `src/culprit/measure.py` | the statistics (step 1) |
| `src/culprit/agent.py` | the agent loop (step 2) |
| `src/culprit/handoff.py` | verified-or-hand-off rules (step 3) |
| `src/culprit/tools/` | the 8 tools |
| `src/culprit/sandbox.py` | command allowlist, worktrees, isolation, path checks |
| `src/culprit/bench_worker.py` | the process that runs and times the benchmark |
| `src/culprit/llm/` | Ollama client and the scripted fake LLM used in tests |
| `perfbench/` | benchmark cases, runner, metrics, failure analysis |
| `results/` | every number, as JSON (nothing in the docs is typed in by hand) |
| `tests/` | unit tests: `make test` |
| `TESTING.md` | checks to run yourself |
