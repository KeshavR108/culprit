#!/bin/bash
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=perfbench/run_suite.py
$PY $R --system agent --variant v3 --model qwen2.5-coder:1.5b --cases all --out results/runs/agent-v3-1.5b
$PY $R --system notools --variant v3 --model qwen2.5-coder:1.5b --cases regressions --out results/runs/notools-v3-1.5b
$PY $R --system agent --variant v3 --model qwen2.5-coder:0.5b --cases all --out results/runs/agent-v3-0.5b
$PY $R --system notools --variant v3 --model qwen2.5-coder:0.5b --cases regressions --out results/runs/notools-v3-0.5b
$PY perfbench/false_alarm.py
echo PHASE5_DONE
