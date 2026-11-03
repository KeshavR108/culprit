"""Phase 2 tests: tools work on a real (tiny) git repo, and the sandbox rules hold."""

import os
import sys

import pytest

from culprit import sandbox
from culprit.sandbox import SandboxError, check_allowed, safe_path
from culprit.session import Config, Session
from culprit.tools import call


@pytest.fixture(scope="module")
def session(toy_repo):
    cfg = Config(repo=toy_repo["repo"], good=toy_repo["shas"][0], bad=toy_repo["shas"][-1],
                 bench="common", bench_file=toy_repo["bench_file"], test_args=["-q"])
    s = Session(cfg)
    yield s
    s.close()


def test_path_traversal_rejected(tmp_path):
    (tmp_path / "ok.py").write_text("x")
    assert safe_path(tmp_path, "ok.py").name == "ok.py"
    for bad in ["../etc/passwd", "/etc/passwd", "a/../../x", ""]:
        with pytest.raises(SandboxError):
            safe_path(tmp_path, bad)


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("secret")
    os.symlink(tmp_path / "secret.txt", root / "link.txt")
    with pytest.raises(SandboxError):
        safe_path(root, "link.txt")


def test_only_allowlisted_commands():
    py = sys.executable
    check_allowed(["git", "log", "-1"], py)
    check_allowed([py, "-m", "pytest", "-q"], py)
    for argv in (["rm", "-rf", "/"], ["bash", "-c", "ls"], [py, "-c", "print(1)"],
                 ["git", "push"], ["git", "reset", "--hard"], ["git", "commit"],
                 ["git", "-C", "/", "log"], ["git", "worktree", "move", "a", "b"], ["curl", "x"]):
        with pytest.raises(SandboxError):
            check_allowed(argv, py)


def test_timeout_enforced(tmp_path):
    (tmp_path / "test_sleep.py").write_text("import time\ndef test_s():\n    time.sleep(30)\n")
    r = sandbox.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=tmp_path,
                    timeout=3, write_root=tmp_path)
    assert r.timed_out


@pytest.mark.skipif(sys.platform != "darwin", reason="OS-level isolation checked on macOS")
def test_no_network_in_sandbox(tmp_path):
    (tmp_path / "test_net.py").write_text(
        "import socket\ndef test_n():\n    socket.create_connection(('1.1.1.1', 80), timeout=3)\n")
    r = sandbox.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=tmp_path,
                    timeout=30, write_root=tmp_path)
    assert r.returncode != 0 and "Operation not permitted" in r.stdout + r.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="OS-level isolation checked on macOS")
def test_no_writes_outside_sandbox_dir(tmp_path):
    outside = tmp_path.parent / "culprit_should_not_exist.txt"
    work = tmp_path / "work"
    work.mkdir()
    (work / "test_w.py").write_text(f"def test_w():\n    open({str(outside)!r}, 'w').write('x')\n")
    r = sandbox.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=work,
                    timeout=30, write_root=work)
    assert r.returncode != 0 and not outside.exists()


def test_run_benchmark_detects_regression(session):
    r = call(session, "run_benchmark", {})
    assert r["verdict"] == "regression" and r["ratio"] > 2


def test_bisect_finds_culprit(session, toy_repo):
    r = call(session, "bisect_commits", {})
    assert r["status"] == "found"
    assert toy_repo["culprit"].startswith(r["first_bad_commit"])
    assert r["steps"] <= 4


def test_profile_points_at_function(session):
    r = call(session, "profile", {})
    assert any("count_common" in f["function"] for f in r["top_on_b"])


def test_show_diff_and_read_file_and_search(session, toy_repo):
    d = call(session, "show_diff", {"commit": toy_repo["culprit"][:8]})
    assert "list(ys)" in d["diff"]
    f = call(session, "read_file", {"path": "src/toy/core.py"})
    assert "lookup = list(ys)" in f["content"]
    assert "error" in call(session, "read_file", {"path": "../../etc/passwd"})
    s = call(session, "search_code", {"pattern": "lookup ="})
    assert any("core.py" in m for m in s["matches"])


def test_bad_arguments_return_errors_not_crashes(session):
    assert "error" in call(session, "show_diff", {})
    assert "error" in call(session, "show_diff", {"commit": "deadbeef"})
    assert "error" in call(session, "nope", {})


def test_propose_and_verify_patch(session):
    bad = call(session, "propose_patch", {"file": "src/toy/core.py", "old_code": "not there",
                                          "new_code": "x"})
    assert bad["valid"] is False
    ok = call(session, "propose_patch", {"file": "src/toy/core.py",
                                         "old_code": "    lookup = list(ys)", "new_code": "    lookup = set(ys)"})
    assert ok["valid"], ok
    v = call(session, "verify_patch", {})
    assert v["verified"] and v["significant_speedup"] and v["tests_passed"], v


def test_patch_that_breaks_tests_is_not_verified(toy_repo):
    cfg = Config(repo=toy_repo["repo"], good=toy_repo["shas"][0], bad=toy_repo["shas"][-1],
                 bench="common", bench_file=toy_repo["bench_file"], test_args=["-q"])
    with Session(cfg) as s:
        r = call(s, "propose_patch", {"file": "src/toy/core.py",
                                      "old_code": "    return sum(1 for x in xs if x in lookup)",
                                      "new_code": "    return 0"})
        assert r["valid"]
        v = call(s, "verify_patch", {})
        assert v["significant_speedup"] and v["tests_passed"] is False and not v["verified"]


def test_tolerant_patch_input_handles_copy_artifacts(tmp_path):
    from culprit.tools import propose_patch as pp
    (tmp_path / "m.py").write_text("def f(xs, ys):\n    lookup = list(ys)\n    return sum(x in lookup for x in xs)\n")
    d = pp.edit_to_diff(tmp_path, "m.py", " def f(xs, ys):\n-    lookup = set(ys)\n+    lookup = list(ys)",
                        " def f(xs, ys):\n+    lookup = set(ys)")
    assert "+    lookup = set(ys)" in d and "-    lookup = list(ys)" in d
    d = pp.edit_to_diff(tmp_path, "m.py", "    2|  lookup = list(ys)", "  lookup = set(ys)")
    assert "+    lookup = set(ys)" in d
    pp.TOLERANT = False
    try:
        d = pp.edit_to_diff(tmp_path, "m.py", "    2|  lookup = list(ys)", "  lookup = set(ys)")
        assert "+     lookup = set(ys)" in d
    finally:
        pp.TOLERANT = True
