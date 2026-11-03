"""PerfBench benchmarks for the HNSW engine."""

from pathlib import Path

import numpy as np

from hnsw import HNSWIndex, load

DATA = Path(__file__).resolve().parent / "data"


def bench_search():
    """200 queries, k=10, default ef, against a prebuilt 1,000-vector index (dim 32)."""
    index = load(DATA / "index_l2.npz")
    Q = np.load(DATA / "queries.npy")
    return lambda: index.batch_search(Q, k=10)


def bench_build():
    """Insert 300 vectors (dim 32) into an empty index that starts with a small buffer."""
    X = np.load(DATA / "build.npy")

    def run():
        idx = HNSWIndex(X.shape[1], M=8, ef_construction=64, seed=0, initial_capacity=16)
        idx.add_batch(X)
    return run
