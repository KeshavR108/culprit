PY := .venv/bin/python
MODEL ?= qwen2.5-coder:3b

.PHONY: setup test test-fast demo gif suite calibrate false-alarm analyze clean-runs

setup:
	python3 -m venv .venv && $(PY) -m pip install -q -e ".[dev]"
	$(PY) perfbench/make_history.py

test:
	$(PY) -m pytest tests -q

test-fast:
	$(PY) -m pytest tests -q -m "not slow"

demo:
	.venv/bin/culprit demo --model $(MODEL)

gif:
	$(PY) scripts/make_demo_gif.py --model $(MODEL)

calibrate:
	$(PY) perfbench/calibrate.py

false-alarm:
	$(PY) perfbench/false_alarm.py

analyze:
	$(PY) perfbench/analyze.py
	$(PY) perfbench/plot_models.py
