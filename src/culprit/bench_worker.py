"""Benchmark worker: runs inside the sandbox, in a checkout of one commit."""

import argparse
import cProfile
import gc
import importlib.util
import json
import os
import pstats
import socket
import sys
import time
import traceback


def block_network():
    """Second layer of 'no network' (the first is the OS sandbox)."""
    def refuse(*a, **k):
        raise OSError("network access is disabled inside the Culprit sandbox")
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-file", required=True)
    p.add_argument("--bench", required=True)
    args = p.parse_args()

    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    def send(obj):
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    if os.environ.get("CULPRIT_NO_NETWORK"):
        block_network()
    try:
        t0 = time.perf_counter()
        spec = importlib.util.spec_from_file_location("culprit_bench", args.bench_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, f"bench_{args.bench}")()
        send({"ready": True, "setup_s": time.perf_counter() - t0})
    except Exception:
        send({"ready": False, "error": traceback.format_exc()[-2000:]})
        return

    cwd = os.path.realpath(os.getcwd())
    for line in sys.stdin:
        msg = json.loads(line)
        try:
            if msg["cmd"] == "run":
                gc.collect()
                t0 = time.perf_counter_ns()
                fn()
                send({"ns": time.perf_counter_ns() - t0})
            elif msg["cmd"] == "profile":
                prof = cProfile.Profile()
                prof.enable()
                for _ in range(int(msg.get("reps", 3))):
                    fn()
                prof.disable()
                stats = pstats.Stats(prof)
                rows = []
                for (file, line_no, func), (cc, nc, tt, ct, _) in stats.stats.items():
                    real = os.path.realpath(file) if not file.startswith("<") else file
                    in_repo = real.startswith(cwd + os.sep)
                    rows.append({"func": func, "file": os.path.relpath(real, cwd) if in_repo else file,
                                 "line": line_no, "in_repo": in_repo, "ncalls": nc,
                                 "tottime": tt, "cumtime": ct})
                rows.sort(key=lambda r: -r["cumtime"])
                send({"rows": rows[:200]})
            elif msg["cmd"] == "quit":
                return
        except Exception:
            send({"error": traceback.format_exc()[-2000:]})


if __name__ == "__main__":
    main()
