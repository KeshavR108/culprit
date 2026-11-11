# TESTING: check Culprit yourself (about 20 minutes)

Everything runs locally. You need Ollama running with `qwen2.5-coder:3b` pulled (`ollama list`).
Run commands from the `Culprit/` folder. Close heavy apps first: timing benchmarks on a busy
laptop makes them noisier (Culprit will say "inconclusive" rather than guess, but it will be slower).

---

## 1. Unit tests

```bash
make test
```

Expected: `30 passed` in about 35 seconds. What they cover:

| file | what it proves |
|---|---|
| `tests/test_measure.py` (7) | identical code → `no_regression` ≥ 95% of 40 checks; a `time.sleep` that doubles runtime → `regression` 20/20; heavy random noise → `inconclusive`, never a false alarm; one huge outlier doesn't fake a regression |
| `tests/test_tools.py` (14) | path traversal and symlink escapes rejected; only allowlisted commands run (`rm`, `bash -c`, `python -c`, `git push`, `git reset` all refused); timeouts enforced; **no network** and **no writes outside the sandbox** (macOS `sandbox-exec`); every tool works on a real tiny git repo; a "fast but wrong" patch is NOT verified because tests fail |
| `tests/test_agent.py` (9) | the agent loop with a scripted fake LLM: happy path verified; wrong commit, low confidence, budget exhausted, invented arguments → all handed off; v2 evidence gate rejects a premature answer |

`make test-fast` skips the real-timing tests (about 20 s).

---

## 2. The demo

```bash
make demo
```

This runs Culprit (v3, 3B model) on a planted regression in your HNSW engine: a commit called
"Simplify visited tracking in search_layer; tidy comments" that changed `visited` from a `set`
to a `list`, hidden among 25 commits. You'll see each step live: the initial measurement, the
LLM's tool calls, bisect, the patch and its verification. Then the full report.

Check the report makes sense:
- Section 1: is the ratio around 1.35× with a CI that excludes 1.05?
- Section 2: did bisect name the "Simplify visited tracking" commit? Look at the bisect table:
  each step is "slow" or "fast" relative to the split point.
- Section 4: is the root cause `src/hnsw/index.py`, near the `visited = list(...)` line?
- With the 3B model this case ends **HANDED OFF** (the model runs out of steps before writing a
  working patch). Read *why* in the report: that's the system working as designed.
- To see a verified fix end to end, run `.venv/bin/culprit demo --case debug_log-1`: one of the 2
  cases (of 36) the 3B agent verifies. It removes a debug-log line from the hot loop; the patched
  copy measures at about 0.87× of the bad commit's time, and the tests pass. Running it again measures everything fresh; the LLM runs
  at temperature 0, so expect the same decisions unless timing noise changes a measurement.

The report and the full step trace are saved under `runs/demo-<time>/`.

---

## 3. Plant your own regression (the real test)

Use a scratch clone so your real repo is never touched:

```bash
git clone https://github.com/keshavr108/semantic-paper-search playground/hnsw
```

**Step 1: break something.** Open `playground/hnsw/src/hnsw/index.py`. In `search_layer`, change

```python
unvisited = [e for e in self.graph[c][layer] if e not in visited]
```

to something slower but still correct. Ideas: make `visited` a `list` (change `set(entry_ids)` to
`list(entry_ids)` and `.update(` to `.extend(`), or wrap the distances call in `_dists` in a
Python loop. Run the tests to make sure it's still correct:

```bash
cd playground/hnsw && PYTHONPATH=src ../../.venv/bin/python -m pytest -q -x -k "not ef200 and not 5k and not hnswlib and not non_decreasing and not better_connected" && cd ../..
```

**Step 2: commit it, then add a harmless commit on top** (so bisect has something to search):

```bash
cd playground/hnsw && git commit -qam "Refactor search_layer" && echo "notes" >> README.md && git commit -qam "Docs" && cd ../..
```

**Step 3: run Culprit** on the last three commits:

```bash
.venv/bin/culprit investigate playground/hnsw --bench search --bench-file perfbench/bench_hnsw.py --good HEAD~2 --bad HEAD --test-args '-q -x -k "not ef200 and not 5k and not hnswlib and not non_decreasing and not better_connected"'
```

Did it find your commit? Your line? Was the fix verified, or did it hand off, and why?
(If your change slows things by less than ~5%, the correct answer is "no regression": that's
the threshold τ at work. Try a bigger slowdown.)

Use `--bench build` instead if you broke something in `add()` / neighbor selection.

---

## 4. A range with NO regression

```bash
.venv/bin/culprit measure perfbench/cases/control-00/repo --bench build --bench-file perfbench/cases/control-00/bench.py --good d884153 --bad 7797145
```

This control case has 31 commits of comments, docs, tests and renames, and no slowdown.
Expected: a ratio near 1.00 and the verdict `no_regression` (or, on a busy laptop,
`inconclusive`, but never `regression`). Run it 2-3 times. The engine's measured false-alarm
rate is in `results/false_alarm.json`.

---

## 5. Check the README numbers against the raw data

```bash
make analyze
```

This recomputes every metric from the per-case JSON files in `results/runs/` and rewrites
`results/metrics.json`, `results/results_table.md` and `results/model_size.png`. Compare
`results/results_table.md` with the table in `README.md`: they must match exactly. Open any
case file, e.g. `results/runs/agent-v3-3b/debug_log-1.json`, and its trace in
`results/runs/agent-v3-3b/traces/debug_log-1.jsonl`, and check by hand that the metric
definitions in `perfbench/analyze.py` (docstring) give the right/wrong call for that case.
