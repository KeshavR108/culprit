"""Catalogue of planted performance regressions for the HNSW engine."""

from dataclasses import dataclass, field


@dataclass
class Edit:
    file: str
    old: str
    new: str


@dataclass
class Regression:
    id: str
    category: str
    bench: str
    message: str
    edits: list[Edit]
    expected: str
    notes: str = ""
    anchor: str = field(default="")


IDX = "src/hnsw/index.py"
DIST = "src/hnsw/distance.py"

REGRESSIONS = [
    Regression(
        "visited_list", "algorithmic", "search", "Simplify visited tracking in search_layer",
        [Edit(IDX, "        visited = set(entry_ids)\n", "        visited = list(entry_ids)\n"),
         Edit(IDX, "            visited.update(unvisited)\n", "            visited.extend(unvisited)\n")],
        "large", "set membership O(1) -> list membership O(n)", anchor="visited = list(entry_ids)"),
    Regression(
        "heap_to_sort", "algorithmic", "search", "Keep result set ordered in search_layer",
        [Edit(IDX,
              "                    heapq.heappush(W, (-d_e, e))\n"
              "                    if len(W) > ef:\n"
              "                        heapq.heappop(W)    # evict the farthest result\n",
              "                    W.append((-d_e, e))\n"
              "                    W.sort()                # farthest result first\n"
              "                    if len(W) > ef:\n"
              "                        W.pop(0)            # evict the farthest result\n")],
        "moderate", "heap push/pop O(log n) -> full sort + list.pop(0) every insert", anchor="W.sort()"),
    Regression(
        "devectorize_dists", "de-vectorization", "search", "Compute distances node by node for readability",
        [Edit(IDX, "        return self._dist_batch(q, self.vectors[ids]).tolist()\n",
              "        return [float(self._dist_batch(q, self.vectors[i:i + 1])[0]) for i in ids]\n")],
        "large", "one NumPy call per batch -> one NumPy call per node",
        anchor="for i in ids]"),
    Regression(
        "devectorize_l2", "de-vectorization", "search", "Write l2 distance as an explicit loop",
        [Edit(DIST, '    return np.einsum("ij,ij->i", diff, diff)\n',
              "    return np.array([np.einsum(\"j,j->\", row, row) for row in diff], dtype=diff.dtype)\n")],
        "large", "einsum over all rows -> Python loop over rows", anchor="for row in diff], dtype=diff.dtype)"),
    Regression(
        "copy_vectors", "copying", "search", "Avoid aliasing the vector storage in _dists",
        [Edit(IDX, "        return self._dist_batch(q, self.vectors[ids]).tolist()\n",
              "        return self._dist_batch(q, self.vectors.copy()[ids]).tolist()\n")],
        "large", "copies the whole vector array on every distance call", anchor="self.vectors.copy()[ids]"),
    Regression(
        "heuristic_recompute", "lost caching", "build", "Inline pairwise distance check in heuristic",
        [Edit(IDX,
              "            row = pair[i]\n"
              "            if all(row[j] > d for j in selected):\n",
              "            if all(self._dist_pairwise(self.vectors[[ids[i], ids[j]]])[0][1] > d for j in selected):\n")],
        "large", "precomputed pairwise matrix ignored; distances recomputed in the loop",
        anchor="self._dist_pairwise(self.vectors[[ids[i], ids[j]]])"),
    Regression(
        "debug_log", "logging in hot path", "search", "Add debug logging to search_layer",
        [Edit(IDX, "import heapq\nimport math\n", "import heapq\nimport logging\nimport math\n"),
         Edit(IDX, "import numpy as np\n\nfrom .distance", "import numpy as np\n\nlog = logging.getLogger(__name__)\n\nfrom .distance"),
         Edit(IDX, "            d_c, c = heapq.heappop(C)\n",
              "            d_c, c = heapq.heappop(C)\n"
              '            log.debug(f"layer {layer}: expand node {c} at distance {d_c:.4f}, |W|={len(W)}")\n')],
        "moderate", "f-string is formatted on every expansion even though DEBUG is off",
        anchor='log.debug(f"layer {layer}: expand node'),
    Regression(
        "prepare_in_loop", "lost caching", "search", "Normalize query defensively at each layer",
        [Edit(IDX, "            ep = [self.search_layer(x, ep, 1, layer)[0][1]]\n        W = self.search_layer(x, ep, ef, 0)[:k]",
              "            ep = [self.search_layer(self._prepare(q), ep, 1, layer)[0][1]]\n        W = self.search_layer(x, ep, ef, 0)[:k]")],
        "subtle", "query re-validated/converted once per layer instead of once",
        anchor="self.search_layer(self._prepare(q), ep, 1, layer)"),
    Regression(
        "sorted_neighbors", "allocation churn", "search", "Visit neighbors in deterministic order",
        [Edit(IDX, "            unvisited = [e for e in self.graph[c][layer] if e not in visited]\n",
              "            unvisited = [e for e in sorted(self.graph[c][layer]) if e not in visited]\n")],
        "subtle", "sorts every neighbor list on every expansion",
        anchor="sorted(self.graph[c][layer])"),
    Regression(
        "ef_default", "configuration", "search", "Raise default ef for better recall",
        [Edit(IDX, "            ef = max(k, 10)\n", "            ef = max(k, 64)\n")],
        "large", "default beam width 10 -> 64: a deliberate-looking trade-off that slows every default search",
        anchor="ef = max(k, 64)"),
    Regression(
        "heapify_each", "algorithmic", "search", "Rebuild candidate heap after each insertion",
        [Edit(IDX, "                    heapq.heappush(C, (d_e, e))\n",
              "                    C.append((d_e, e))\n                    heapq.heapify(C)\n")],
        "moderate", "O(log n) push -> O(n) heapify per insertion", anchor="                    heapq.heapify(C)"),
    Regression(
        "pairwise_python", "de-vectorization", "build", "Write pairwise l2 as an explicit double loop",
        [Edit(DIST,
              "    sq = np.einsum(\"ij,ij->i\", X, X)\n"
              "    with _quiet_matmul():\n"
              "        D = sq[:, None] - 2.0 * (X @ X.T) + sq[None, :]\n"
              "    return np.maximum(D, 0.0)\n",
              "    return np.array([[l2_sq(a, b) for b in X] for a in X], dtype=np.float32)\n")],
        "large", "one matrix multiply -> n^2 Python-level distance calls",
        anchor="[[l2_sq(a, b) for b in X] for a in X]"),
    Regression(
        "metric_lookup", "lost caching", "search", "Look up the metric function at call time",
        [Edit(IDX, "        return self._dist_batch(q, self.vectors[ids]).tolist()\n",
              "        return get_metric(self.metric)[0](q, self.vectors[ids]).tolist()\n")],
        "subtle", "cached function pointer -> name lookup + branch on every distance call",
        anchor="get_metric(self.metric)[0](q"),
    Regression(
        "stop_check_scan", "algorithmic", "search", "Make the search stop condition explicit",
        [Edit(IDX, "            if d_c > -W[0][0] and len(W) >= ef:\n",
              "            if len(W) >= ef and d_c > max(-w for w, _ in W):\n")],
        "moderate", "O(1) heap peek -> O(ef) scan of the result set on every pop (same results)",
        anchor="max(-w for w, _ in W)"),
    Regression(
        "always_reprune", "algorithmic", "build", "Keep reverse links well-chosen on every insert",
        [Edit(IDX, "                if len(links) > Mmax:                # e is over capacity: re-select\n",
              "                if len(links) > 1:                   # re-select so links stay diverse\n")],
        "moderate", "neighbor re-selection (distances + pairwise matrix) runs on every reverse link, not only over capacity",
        anchor="if len(links) > 1:"),
    Regression(
        "wide_descent", "configuration", "search", "Use the full beam on upper layers too",
        [Edit(IDX, "            ep = [self.search_layer(x, ep, 1, layer)[0][1]]\n        W = self.search_layer(x, ep, ef, 0)[:k]",
              "            ep = [self.search_layer(x, ep, ef, layer)[0][1]]\n        W = self.search_layer(x, ep, ef, 0)[:k]")],
        "moderate", "greedy (ef=1) descent through upper layers -> beam search with ef on every layer",
        anchor="ep = [self.search_layer(x, ep, ef, layer)[0][1]]"),
    Regression(
        "stack_gather", "allocation churn", "search", "Gather vectors explicitly in _dists",
        [Edit(IDX, "        return self._dist_batch(q, self.vectors[ids]).tolist()\n",
              "        return self._dist_batch(q, np.stack([self.vectors[i] for i in ids])).tolist()\n")],
        "moderate", "one fancy-indexing gather -> Python loop building a list of row views, then np.stack",
        anchor="np.stack([self.vectors[i] for i in ids])"),
]

BY_ID = {r.id: r for r in REGRESSIONS}
