"""Sandbox: everything Culprit executes goes through here."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

WORKER = Path(__file__).with_name("bench_worker.py")

GIT_ALLOWED = {"rev-parse", "rev-list", "log", "show", "diff", "ls-files", "cat-file",
               "worktree", "apply", "merge-base"}
GIT_WORKTREE_ALLOWED = {"add", "remove", "prune", "list"}


class SandboxError(Exception):
    """A command or path was rejected by the sandbox rules."""


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def safe_path(root: Path, rel: str) -> Path:
    """Resolve `rel` inside `root`; reject anything that escapes it (../, absolute, symlinks)."""
    root = Path(root).resolve()
    if not rel or "\x00" in rel:
        raise SandboxError("empty or invalid path")
    p = Path(rel)
    if p.is_absolute():
        raise SandboxError(f"absolute paths are not allowed: {rel}")
    full = (root / p).resolve()
    if not full.is_relative_to(root):
        raise SandboxError(f"path escapes the repository: {rel}")
    return full


def _sandbox_profile(write_root: Path | None) -> str:
    """macOS seatbelt profile: deny network; optionally deny writes outside write_root."""
    rules = ["(version 1)", "(allow default)", "(deny network*)",
             '(allow network* (local unix-socket))']
    if write_root is not None:
        rules += ["(deny file-write*)",
                  f'(allow file-write* (subpath "{Path(write_root).resolve()}"))',
                  '(allow file-write* (literal "/dev/null"))',
                  '(allow file-write* (regex #"^/dev/tty"))',
                  '(allow file-write* (regex #"^/dev/fd/"))']
    return "".join(rules)


def isolate(argv: list[str], write_root: Path | None = None) -> list[str]:
    """Wrap argv so it runs without network (and with restricted writes when possible)."""
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return ["sandbox-exec", "-p", _sandbox_profile(write_root), *argv]
    if sys.platform.startswith("linux") and shutil.which("unshare"):
        return ["unshare", "-rn", *argv]
    return argv


def check_allowed(argv: list[str], python: str) -> None:
    """Raise SandboxError unless argv is on the allowlist."""
    if not argv:
        raise SandboxError("empty command")
    exe = argv[0]
    if exe == "git":
        sub = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else None
        if sub == "-C" or (len(argv) > 2 and argv[1] == "-C"):
            raise SandboxError("use cwd instead of git -C")
        if sub not in GIT_ALLOWED:
            raise SandboxError(f"git subcommand not allowed: {sub}")
        if sub == "worktree" and (len(argv) < 3 or argv[2] not in GIT_WORKTREE_ALLOWED):
            raise SandboxError("git worktree subcommand not allowed")
        return
    if exe == python:
        if argv[1:3] == ["-m", "pytest"]:
            return
        if len(argv) > 1 and Path(argv[1]).resolve() == WORKER.resolve():
            return
        raise SandboxError("python may only run pytest or the Culprit benchmark worker")
    raise SandboxError(f"command not allowed: {exe}")


def run(argv: list[str], cwd: Path, python: str = sys.executable, timeout: float = 60,
        write_root: Path | None = None, env: dict | None = None) -> Result:
    """Run an allowlisted command with a timeout and no network."""
    check_allowed(argv, python)
    full_env = {**os.environ, **(env or {}), "PYTHONDONTWRITEBYTECODE": "1",
                "GIT_TERMINAL_PROMPT": "0",
                "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9"}
    if write_root is not None:
        full_env["TMPDIR"] = str(write_root)
    try:
        p = subprocess.run(isolate(argv, write_root), cwd=cwd, capture_output=True,
                           text=True, timeout=timeout, env=full_env)
        return Result(p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired as e:
        return Result(-1, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                      f"timed out after {timeout}s", timed_out=True)


def git(repo: Path, *args: str, timeout: float = 60) -> str:
    """Run an allowlisted git command in `repo` and return stdout (raises on failure)."""
    r = run(["git", *args], cwd=repo, timeout=timeout)
    if r.returncode != 0:
        raise SandboxError(f"git {' '.join(args)} failed: {r.stderr.strip()[:500]}")
    return r.stdout


class Worktrees:
    """Temporary checkouts of commits, created with `git worktree add` in a temp dir."""

    def __init__(self, repo: Path):
        self.repo = Path(repo).resolve()
        self.root = Path(tempfile.mkdtemp(prefix="culprit-")).resolve()
        self._made: dict[str, Path] = {}
        self._n = 0

    def get(self, sha: str) -> Path:
        """A shared read-only checkout of `sha` (reused across tools)."""
        if sha not in self._made:
            self._made[sha] = self._add(sha, f"wt-{sha[:12]}")
        return self._made[sha]

    def fresh(self, sha: str, tag: str = "scratch") -> Path:
        """A new private checkout of `sha` that may be modified (e.g."""
        self._n += 1
        return self._add(sha, f"wt-{sha[:12]}-{tag}-{self._n}")

    def _add(self, sha: str, name: str) -> Path:
        path = self.root / name
        git(self.repo, "worktree", "add", "--detach", "--quiet", str(path), sha, timeout=120)
        return path

    def cleanup(self) -> None:
        for path in self.root.glob("wt-*"):
            run(["git", "worktree", "remove", "--force", str(path)], cwd=self.repo, timeout=60)
        run(["git", "worktree", "prune"], cwd=self.repo, timeout=60)
        shutil.rmtree(self.root, ignore_errors=True)


class BenchWorker:
    """A long-lived Python process holding one commit's benchmark, ready to run it."""

    def __init__(self, checkout: Path, bench_file: Path, bench: str, python: str,
                 write_root: Path, setup_timeout: float = 180, run_timeout: float = 120):
        self.run_timeout = run_timeout
        argv = [python, str(WORKER), "--bench-file", str(bench_file), "--bench", bench]
        check_allowed(argv, python)
        src = checkout / "src"
        pythonpath = os.pathsep.join(str(p) for p in (src, checkout) if p.exists())
        env = {**os.environ, "PYTHONPATH": pythonpath, "PYTHONDONTWRITEBYTECODE": "1",
               "TMPDIR": str(write_root), "CULPRIT_NO_NETWORK": "1"}
        self.log_path = Path(write_root) / f"worker-{checkout.name}.log"
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(isolate(argv, write_root), cwd=checkout, env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._log, text=True, bufsize=1)
        self.info = self._recv(setup_timeout)
        if not self.info.get("ready"):
            raise SandboxError(f"benchmark setup failed: {self.info.get('error', '?')}")

    def _send(self, msg: dict) -> None:
        import json
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _recv(self, timeout: float) -> dict:
        import json
        import selectors
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        if not sel.select(timeout):
            self.close()
            raise SandboxError(f"benchmark worker timed out after {timeout}s")
        line = self.proc.stdout.readline()
        if not line:
            self._log.flush()
            err = self.log_path.read_text(errors="replace")[-1500:]
            raise SandboxError(f"benchmark worker exited: {err}")
        return json.loads(line)

    def run_once(self) -> int:
        """Run the benchmark once; return wall time in ns (timed inside the worker)."""
        self._send({"cmd": "run"})
        r = self._recv(self.run_timeout)
        if "error" in r:
            raise SandboxError(f"benchmark raised: {r['error']}")
        return r["ns"]

    def profile(self, reps: int = 3) -> list[dict]:
        self._send({"cmd": "profile", "reps": reps})
        r = self._recv(self.run_timeout * reps)
        if "error" in r:
            raise SandboxError(f"profiling failed: {r['error']}")
        return r["rows"]

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send({"cmd": "quit"})
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self._log.close()
