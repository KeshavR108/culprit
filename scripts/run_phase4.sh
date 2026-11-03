#!/bin/bash
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY perfbench/run_suite.py --system classic --cases all --out results/runs/classic
$PY perfbench/run_suite.py --system agent --variant v1 --model qwen2.5-coder:3b --cases all --out results/runs/agent-v1-3b
$PY perfbench/run_suite.py --system notools --model qwen2.5-coder:3b --cases regressions --out results/runs/notools-3b
echo PHASE4_DONE
