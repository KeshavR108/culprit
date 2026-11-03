"""Command line: culprit investigate <repo> --bench <name> --bench-file <file> --good A --bad B"""

from __future__ import annotations

import os
import shlex
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from . import prompts
from . import report as report_mod
from .agent import investigate as run_agent
from .llm.ollama_client import OllamaLLM, available_models
from .session import Config, Session

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False, help="Find out why your code got slower, with statistics.")
console = Console()
ROOT = Path(__file__).resolve().parents[2]


def _cfg(repo, bench, bench_file, good, bad, python, test_args, tau) -> Config:
    return Config(repo=Path(repo), good=good, bad=bad, bench=bench, bench_file=Path(bench_file),
                  python=python or sys.executable, test_args=shlex.split(test_args), tau=tau)


@app.command()
def measure(repo: str, bench: str = typer.Option(...), bench_file: str = typer.Option(...),
            good: str = typer.Option(...), bad: str = typer.Option(...), tau: float = 0.05,
            python: str = typer.Option(None)):
    """Only the deterministic measurement: is BAD really slower than GOOD?"""
    cfg = _cfg(repo, bench, bench_file, good, bad, python, "-q", tau)
    with Session(cfg) as s:
        m = s.compare(s.good, s.bad)
    console.print(m.summary())
    console.print(f"[bold]{m.verdict}[/bold]")


@app.command()
def investigate(repo: str, bench: str = typer.Option(...), bench_file: str = typer.Option(...),
                good: str = typer.Option(...), bad: str = typer.Option(...),
                model: str = "qwen2.5-coder:3b", test_args: str = "-q -x", tau: float = 0.05,
                python: str = typer.Option(None, help="interpreter with the target's dependencies"),
                out: str = typer.Option(None, help="directory for report + trace (default runs/<time>)"),
                apply: bool = typer.Option(False, "--apply", help="offer to apply a verified fix"),
                variant: str = typer.Option("v3", help="agent version: v1 baseline, v2 +evidence gate, v3 +tolerant patches")):
    """Measure, bisect, profile, diagnose, propose and verify a fix (or hand off)."""
    from .tools import propose_patch
    prompts.set_variant(variant)
    propose_patch.TOLERANT = variant not in ("v1", "v2")
    if model not in available_models():
        console.print(f"[red]Model {model} is not pulled in Ollama. Run: ollama pull {model}[/red]")
        raise typer.Exit(1)
    out_dir = Path(out or ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(repo, bench, bench_file, good, bad, python, test_args, tau)
    console.print(f"[bold]Culprit[/bold]: investigating `{bench}` with {model} (local, via Ollama)")
    rep = run_agent(cfg, OllamaLLM(model), out_dir / "trace.jsonl", echo=_progress)
    md = report_mod.to_markdown(rep)
    (out_dir / "report.md").write_text(md)
    (out_dir / "report.json").write_text(rep.model_dump_json(indent=2))
    console.print(Markdown(md))
    console.print(f"\nSaved: {out_dir / 'report.md'}")
    if apply and rep.status == "verified" and rep.patch:
        _apply(Path(repo), rep.patch.diff)


def _progress(e: dict) -> None:
    """One short line per trace event, so the user can watch the investigation."""
    t = f"[dim]{e['t']:6.1f}s[/dim]"
    k = e["kind"]
    if k == "measurement":
        console.print(f"{t} measure  {e['summary']}" + (" [dim](cached)[/dim]" if e.get("cached") else ""))
    elif k == "llm":
        calls = ", ".join(f"{c['name']}({', '.join(f'{a}={v!r}'[:40] for a, v in c['arguments'].items())})"
                          for c in e["tool_calls"])
        console.print(f"{t} [cyan]LLM step {e['step']}[/cyan] -> " + (calls or "final answer"))
    elif k == "tool":
        console.print(f"{t}   [green]{e['name']}[/green] -> {e['result'][:110]}")
    elif k == "verification":
        console.print(f"{t} verify   verified={e['verified']} tests_passed={e['tests_passed']}")
    elif k == "report":
        console.print(f"{t} [bold]{e['status'].upper()}[/bold] {'; '.join(e['reasons'])}\n")


def _apply(repo: Path, diff: str) -> None:
    """Applying to the real repo needs --apply AND typing 'apply'."""
    console.print("[yellow]This will modify files in your working tree (no commit).[/yellow]")
    if input("Type 'apply' to apply the verified patch: ").strip() != "apply":
        console.print("Not applied.")
        return
    import subprocess
    p = subprocess.run(["git", "apply", "--recount", "-"], cwd=repo, input=diff, text=True,
                       capture_output=True)
    console.print("Applied." if p.returncode == 0 else f"[red]git apply failed: {p.stderr}[/red]")


@app.command()
def demo(model: str = "qwen2.5-coder:3b", case: str = typer.Option("demo", help="PerfBench case id")):
    """Run Culprit on a planted regression in the HNSW engine and print the report."""
    sys.path.insert(0, str(ROOT))
    from perfbench import make_history
    case_dir = make_history.ensure_demo_case(case)
    meta = make_history.load_meta(case_dir)
    investigate(str(case_dir / "repo"), bench=meta["bench"], bench_file=str(case_dir / "bench.py"),
                good=meta["good"], bad=meta["bad"], model=model, test_args=meta["test_args"],
                tau=0.05, python=None, out=str(ROOT / "runs" / f"demo-{time.strftime('%Y%m%d-%H%M%S')}"),
                apply=False, variant="v3")


if __name__ == "__main__":
    app()
