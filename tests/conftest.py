"""A tiny git repo with a planted regression, used by the tool and agent tests."""

import subprocess
import textwrap
from pathlib import Path

import pytest

FAST = textwrap.dedent('''\
    def count_common(xs, ys):
        """How many items of xs are also in ys."""
        lookup = set(ys)
        return sum(1 for x in xs if x in lookup)
    ''')
SLOW = FAST.replace("lookup = set(ys)", "lookup = list(ys)")

BENCH = textwrap.dedent('''\
    from toy import count_common

    def bench_common():
        xs = list(range(60000))          # ~3 ms with a set, ~10x slower with a list
        ys = list(range(0, 120000, 2000))
        return lambda: count_common(xs, ys)
    ''')

TEST = textwrap.dedent('''\
    from toy import count_common

    def test_count_common():
        assert count_common([1, 2, 3], [2, 3, 4]) == 2
    ''')


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def commit(repo, msg):
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture(scope="session")
def toy_repo(tmp_path_factory):
    """History: c0 (good) -> c1 benign -> c2 REGRESSION -> c3 benign -> c4 benign (bad)."""
    base = tmp_path_factory.mktemp("toy")
    repo = base / "repo"
    (repo / "src" / "toy").mkdir(parents=True)
    (repo / "tests").mkdir()
    git(repo, "init", "-q")
    (repo / "src" / "toy" / "__init__.py").write_text("from .core import count_common\n")
    core = repo / "src" / "toy" / "core.py"
    core.write_text(FAST)
    (repo / "tests" / "test_toy.py").write_text(TEST)
    shas = [commit(repo, "initial")]
    (repo / "README.md").write_text("toy\n")
    shas.append(commit(repo, "Add README"))
    core.write_text(SLOW)
    shas.append(commit(repo, "Simplify lookup"))
    (repo / "README.md").write_text("toy package\n")
    shas.append(commit(repo, "Docs"))
    core.write_text("# core helpers\n" + SLOW)
    shas.append(commit(repo, "Header comment"))
    bench = base / "bench_toy.py"
    bench.write_text(BENCH)
    return {"repo": repo, "bench_file": bench, "shas": shas, "culprit": shas[2]}
