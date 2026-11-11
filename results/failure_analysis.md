# Failure analysis (qwen2.5-coder:3b)

Generated from `results/failures/<run>.json` (made by `perfbench/failures.py`, which classifies every
miss from its trace) plus hand-picked real examples. A miss is a detectable regression that did not end
"verified" with the right diagnosis. Each miss gets the EARLIEST thing that went wrong.

Not on the list: wrong bisect (0 in every run: bisect found the true culprit commit in 36/36 cases)
and false fixes (0 in every run).

## Counts by version

| failure | v1 | v2 | v3 |
|---|---|---|---|
| answered before looking at code | **36** | 0 | 0 |
| looped without progress | 0 | **20** | 10 |
| ran out of steps | 0 | 2 | 2 |
| right commit, wrong line | 0 | 1 | 2 |
| patch failed to run | 0 | 5 | 8 |
| patch didn't help | 0 | 7 | **10** |
| patch broke tests | 0 | 0 | 2 |
| **total misses (of 36)** | 36 | 35 | 34 |

Model-size runs (v3): see `results/failures/agent-v3-1.5b.json` and `agent-v3-0.5b.json`.

## One real example per category

**Answered before looking at code (v1, 36/36).** The model called `bisect_commits` (correct commit), then
immediately answered `{"file": "search.py", "line": 100, ...}`. That file doesn't exist; the code is in
`src/hnsw/index.py`. 21 of the 36 answers named `search.py:100` or `src/search.py:100`.

**Looped without progress (v2: 20, v3: 10).** `always_reprune-0` (v3): `propose_patch` called 11 times with
identical arguments after the same "old_code was not found" error. In v2 the usual cause was copied diff
markers (`-`, `+`) and `210|` line-number prefixes inside `old_code`; v3 cleans those up, which halved this
category. The remaining loops are mostly `old_code` that doesn't exist in the file at all (the model writes
what it *thinks* the code says).

**Patch didn't help (v3: 10).** `devectorize_dists-2`: the explanation is correct ("computes distances for
each node in `ids` individually, which is more expensive than computing them in batches"), but the patch
pastes the slow line into a different function instead of restoring the batched call. Verification
measured bad → patched at 0.99× and refused it. **Right diagnosis, wrong fix** is the most common pattern here.

**Patch failed to run (v3: 8).** e.g. `copy_vectors-1`: the patched copy raised during benchmark setup
(syntax or indentation errors in model-written code). Caught by `verify_patch` before any tests ran.

**Patch broke tests (v3: 2).** `always_reprune-2`: the model claimed the slowdown came from a new
membership check, and "fixed" it with `links = self.graph[e][layer] if e in self.graph else []`.
`self.graph` is a list, so `e in self.graph` is always False and every reverse link is silently dropped.
The tests failed (`assert 2 == 3`) and it was not verified. This is the hand-off rules doing their job:
a fast, harmful patch was rejected.

**Right commit, wrong line (v3: 2).** `always_reprune-1`: claimed `index.py:175`, planted change at line 218.

**Ran out of steps (v3: 2).** `copy_vectors-0`: alternated `propose_patch` and `read_file` until step 15.

## The targeted improvements

1. **v1 → v2, evidence gate** (deterministic, in `agent.py`). Target: "answered before looking at code"
   (36/36). The loop refuses a final answer until the agent has looked at code, proposed a patch and verified
   it, and tells the model the specific next step. Result: that category went to 0. Root-cause line (±5) went
   0% → 36%, verified fixes 0 → 1. The failure moved downstream: looping on patch input.
2. **v2 → v3, tolerant patch input** (deterministic, in `propose_patch.py`). Target: "looped without
   progress" (20/35). It cleans diff markers, line-number prefixes and indentation, and finds the ONE matching
   location (ambiguous matches are still rejected). It also fixes a v2 bug where a mid-line match silently
   produced a mis-indented patch. Result: loops 20 → 10, root-cause line (±5) 36% → 61%, verified fixes 1 → 2.

Both changes are deterministic code, not prompt tweaks. What remains is mostly the model's reasoning about
*what the fix is*: writing a correct patch for a correctly understood problem. That's where a bigger model
is the next lever (see the model-size study).

## Known issues found along the way (not yet fixed)

- When a model's tool call is malformed (e.g. truncated JSON), the evidence gate replies "you have not proposed
  a fix yet", which is misleading. It should say "your tool call was malformed". Seen mostly at 1.5B.
- The ±5-line tolerance gives credit for citing a diff hunk's start line; the stricter ±1 metric is reported
  alongside it for that reason.
