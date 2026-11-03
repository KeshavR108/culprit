"""Build PerfBench: git histories of the HNSW engine with a regression hidden among benign commits."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from regressions import BY_ID, REGRESSIONS, Regression

TARGET_REPO = ROOT / "targets" / "semantic-paper-search"
TARGET_URL = "https://github.com/keshavr108/semantic-paper-search"
TARGET_COMMIT = "5cc47e0"
CASES = HERE / "cases"
DATA = HERE / "data"
BASE_FILES = ["src/hnsw/__init__.py", "src/hnsw/index.py", "src/hnsw/distance.py", "src/hnsw/brute_force.py",
              "src/hnsw/io.py", "tests/test_distance.py", "tests/test_graph_invariants.py", "tests/test_io.py",
              "tests/test_levels.py", "tests/test_recall.py", "tests/test_search_layer.py",
              "tests/test_select_neighbors.py", "pyproject.toml", "README.md", ".gitignore"]
FAST_TESTS = '-q -x -k "not ef200 and not 5k and not hnswlib and not non_decreasing and not better_connected"'
SRC_FILES = ["src/hnsw/index.py", "src/hnsw/distance.py", "src/hnsw/brute_force.py", "src/hnsw/io.py"]


def base_files() -> dict[str, str]:
    if not TARGET_REPO.exists():
        subprocess.run(["git", "clone", "-q", TARGET_URL, str(TARGET_REPO)], check=True)
    return {f: subprocess.run(["git", "show", f"{TARGET_COMMIT}:{f}"], cwd=TARGET_REPO, check=True,
                              capture_output=True, text=True).stdout for f in BASE_FILES}


def ensure_data() -> None:
    """Benchmark data + a prebuilt index, made once with the BASE code (seeded)."""
    if (DATA / "index_l2.npz").exists():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = HERE / ".base_src"
    shutil.rmtree(tmp, ignore_errors=True)
    for f, text in base_files().items():
        (tmp / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp / f).write_text(text)
    sys.path.insert(0, str(tmp / "src"))
    from hnsw import HNSWIndex, save
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 32)).astype(np.float32)
    Q = rng.standard_normal((200, 32)).astype(np.float32)
    idx = HNSWIndex(32, M=8, ef_construction=64, seed=0)
    idx.add_batch(X)
    save(idx, DATA / "index_l2.npz")
    np.save(DATA / "queries.npy", Q)
    np.save(DATA / "build.npy", X[:300])
    sys.path.remove(str(tmp / "src"))
    shutil.rmtree(tmp)


COMMENTS = ["# See README for how the parameters interact.", "# Kept deliberately simple; profile before optimizing.",
            "# Covered by tests in tests/.", "# Public API: keep the signature stable.",
            "# Paper reference: Malkov & Yashunin, arXiv:1603.09320.", "# TODO: add a C++ fast path later."]
DOC_SENTENCES = ["All functions here are pure: they never modify their inputs.",
                 "Shapes follow NumPy conventions: rows are vectors.",
                 "See tests/ for executable examples of every public function.",
                 "Distances are always 'lower is closer'."]


def e_comment_above_def(files, rng, st, f=None):
    f = f or rng.choice(SRC_FILES)
    lines = files[f].splitlines(True)
    defs = [i for i, l in enumerate(lines) if l.lstrip().startswith("def ") and not lines[i - 1].lstrip().startswith("@")]
    i = rng.choice(defs)
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    name = lines[i].split("def ")[1].split("(")[0]
    lines.insert(i, indent + rng.choice(COMMENTS) + "\n")
    return {**files, f: "".join(lines)}, f"Add comment above {name}"


def e_module_doc(files, rng, st, f=None):
    f = f or rng.choice(SRC_FILES)
    text = files[f]
    end = text.find('"""', 3)
    if not text.startswith('"""') or end < 0:
        return None
    s = rng.choice(DOC_SENTENCES)
    if s in text:
        return None
    return {**files, f: text[:end].rstrip() + "\n\n" + s + "\n" + text[end:]}, f"Docs: clarify {Path(f).stem} module docstring"


def e_add_test(files, rng, st):
    st["tests"] = st.get("tests", 0) + 1
    k, n = st["tests"], rng.randint(10, 30)
    body = (f'import numpy as np\n\nfrom hnsw import HNSWIndex\n\n\n'
            f'def test_len_after_{n}_adds_{k}():\n'
            f'    rng = np.random.default_rng({k})\n'
            f'    idx = HNSWIndex(4, M=4, ef_construction=16, seed={k})\n'
            f'    idx.add_batch(rng.standard_normal(({n}, 4)))\n'
            f'    assert len(idx) == {n}\n'
            f'    ids, d = idx.search(rng.standard_normal(4), k=3)\n'
            f'    assert len(ids) == 3 and list(d) == sorted(d)\n')
    return {**files, f"tests/test_extra_{k}.py": body}, f"Add small-index test #{k}"


def e_readme(files, rng, st):
    st["readme"] = st.get("readme", 0) + 1
    line = rng.choice(["Run the tests with `pytest -q`.", "Benchmarks live in `bench/`.",
                       "Contributions welcome: open an issue first.", "Python 3.11+ is required."])
    return {**files, "README.md": files["README.md"].rstrip() + f"\n\n{line}\n"}, "README: add usage note"


def e_changelog(files, rng, st):
    st["cl"] = st.get("cl", 0) + 1
    old = files.get("CHANGELOG.md", "# Changelog\n")
    return {**files, "CHANGELOG.md": old + f"\n- Maintenance update {st['cl']}\n"}, "Update CHANGELOG"


def e_helper(files, rng, st):
    st["helper"] = st.get("helper", 0) + 1
    f = rng.choice(["src/hnsw/brute_force.py", "src/hnsw/io.py", "src/hnsw/distance.py"])
    k = st["helper"]
    fn = (f'\n\ndef _shape_str_{k}(X) -> str:\n    """Human-readable shape, for error messages."""\n'
          f'    return "x".join(str(n) for n in np.shape(X))\n')
    return {**files, f: files[f].rstrip() + "\n" + fn}, f"Add shape helper to {Path(f).stem}"


def e_repr(files, rng, st):
    f = "src/hnsw/index.py"
    anchor = "    def __len__(self) -> int:\n        return self.count\n"
    if "__repr__" in files[f] or anchor not in files[f]:
        return None
    rep = (anchor + "\n    def __repr__(self) -> str:\n"
           '        return f"HNSWIndex(dim={self.dim}, metric={self.metric!r}, count={self.count}, M={self.M})"\n')
    return {**files, f: files[f].replace(anchor, rep)}, "Add __repr__ for easier debugging"


def e_rename_io(files, rng, st):
    f = "src/hnsw/io.py"
    if "degree_list" in files[f]:
        return None
    text = files[f]
    for a, b in [("    degrees = [len(nbrs)", "    degree_list = [len(nbrs)"),
                 ("np.array(degrees, dtype", "np.array(degree_list, dtype"),
                 ('        degrees = f["degrees"]', '        degree_list = f["degrees"]'),
                 ("degrees[slot]", "degree_list[slot]")]:
        text = text.replace(a, b)
    return {**files, f: text}, "Rename degrees -> degree_list in io"


def e_version(files, rng, st):
    st["ver"] = st.get("ver", 0) + 1
    import re
    new = re.sub(r'version = "0\.1\.\d+"', f'version = "0.1.{st["ver"]}"', files["pyproject.toml"])
    return {**files, "pyproject.toml": new}, f"Bump version to 0.1.{st['ver']}"


def e_gitignore(files, rng, st):
    entry = rng.choice(["*.npz", ".DS_Store", "*.log", "build/", "dist/"])
    if entry in files[".gitignore"]:
        return None
    return {**files, ".gitignore": files[".gitignore"].rstrip() + f"\n{entry}\n"}, f"Ignore {entry}"


BENIGN = [e_comment_above_def, e_comment_above_def, e_module_doc, e_add_test, e_add_test, e_readme,
          e_changelog, e_helper, e_repr, e_rename_io, e_version, e_gitignore]


def apply_regression(files, reg: Regression) -> dict:
    out = dict(files)
    for e in reg.edits:
        assert out[e.file].count(e.old) == 1, f"{reg.id}: anchor not unique in {e.file}"
        out[e.file] = out[e.file].replace(e.old, e.new)
    return out


def anchors_intact(files, reg: Regression | None) -> bool:
    return reg is None or all(files[e.file].count(e.old) == 1 for e in reg.edits)


def planted_lines(files, reg: Regression) -> dict:
    """Ground truth: the file and the line numbers of the slow code (1-based)."""
    f = reg.edits[-1].file
    text = files[f]
    i = text.index(reg.anchor)
    first = text.count("\n", 0, i) + 1
    lines = set()
    for e in reg.edits:
        if e.file == f:
            j = text.index(e.new)
            start = text.count("\n", 0, j) + 1
            lines.update(range(start, start + max(1, e.new.rstrip("\n").count("\n") + 1)))
    return {"file": f, "anchor_line": first, "lines": sorted(lines)}


def git(repo, *args, env=None):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env={**os.environ, **(env or {})}).stdout.strip()


def write_files(repo: Path, files: dict) -> None:
    for f, text in files.items():
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text(text)


def commit(repo: Path, msg: str, n: int) -> str:
    date = f"2026-09-01T{10 + n // 60:02d}:{n % 60:02d}:00"
    env = {"GIT_AUTHOR_NAME": "PerfBench", "GIT_AUTHOR_EMAIL": "perfbench@example.com",
           "GIT_COMMITTER_NAME": "PerfBench", "GIT_COMMITTER_EMAIL": "perfbench@example.com",
           "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg, env=env)
    return git(repo, "rev-parse", "HEAD")


def make_case(case_id: str, reg_id: str | None, seed: int, bench: str | None = None) -> Path:
    """Create one case directory: repo/ (git history), bench.py, data/, meta.json."""
    rng = random.Random(seed)
    reg = BY_ID[reg_id] if reg_id else None
    bench = reg.bench if reg else (bench or rng.choice(["search", "build"]))
    n_after = rng.randint(15, 31)
    pos = rng.randrange(n_after) if reg else None
    mixed = bool(reg) and rng.random() < 0.5

    case = CASES / case_id
    shutil.rmtree(case, ignore_errors=True)
    repo = case / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    files = base_files()
    write_files(repo, files)
    shas = [commit(repo, "Initial import of HNSW engine", 0)]
    st, culprit, planted_culprit, culprit_msg = {}, None, None, None

    for i in range(n_after):
        if reg and i == pos:
            msg = reg.message
            if mixed:
                for _ in range(20):
                    target = reg.edits[-1].file
                    out = (e_comment_above_def(files, rng, st, target) if rng.random() < 0.7
                           else e_module_doc(files, rng, st, target))
                    if out and anchors_intact(out[0], reg):
                        files = out[0]
                        msg += "; tidy comments"
                        break
            files = apply_regression(files, reg)
            planted_culprit = planted_lines(files, reg)
            write_files(repo, files)
            culprit = commit(repo, msg, i + 1)
            culprit_msg = msg
            shas.append(culprit)
            continue
        for _ in range(50):
            out = rng.choice(BENIGN)(files, rng, st)
            if out is None:
                continue
            if culprit is None and anchors_intact(out[0], reg):
                break
            if culprit is not None and out[0][reg.edits[-1].file].count(reg.anchor) == 1:
                break
        else:
            raise RuntimeError(f"{case_id}: no benign edit applies")
        files, msg = out
        write_files(repo, files)
        shas.append(commit(repo, msg, i + 1))

    shutil.copy(HERE / "bench_hnsw.py", case / "bench.py")
    shutil.copytree(DATA, case / "data")
    meta = {"id": case_id, "kind": "regression" if reg else "control", "seed": seed, "bench": bench,
            "good": shas[0], "bad": shas[-1], "n_commits_after_good": n_after, "test_args": FAST_TESTS,
            "target": {"repo": TARGET_URL, "commit": TARGET_COMMIT}}
    if reg:
        meta.update({"regression": reg.id, "category": reg.category, "expected": reg.expected,
                     "culprit": culprit, "culprit_message": culprit_msg, "position": pos, "mixed": mixed,
                     "planted_culprit": planted_culprit, "planted_head": planted_lines(files, reg)})
    (case / "meta.json").write_text(json.dumps(meta, indent=2))
    return case


def load_meta(case_dir: Path) -> dict:
    return json.loads((Path(case_dir) / "meta.json").read_text())


def suite_spec(variants: int = 3, controls: int = 12) -> list[tuple[str, str | None, int]]:
    spec = []
    for k, reg in enumerate(REGRESSIONS):
        for v in range(variants):
            spec.append((f"{reg.id}-{v}", reg.id, 1000 * k + v))
    for c in range(controls):
        spec.append((f"control-{c:02d}", None, 90_000 + c))
    return spec


def ensure_demo_case(case_id: str = "demo") -> Path:
    ensure_data()
    case = CASES / case_id
    if not (case / "meta.json").exists():
        make_case(case_id, "visited_list", seed=7)
    return case


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--controls", type=int, default=12)
    args = p.parse_args()
    ensure_data()
    spec = suite_spec(args.variants, args.controls)
    for case_id, reg_id, seed in spec:
        make_case(case_id, reg_id, seed)
    make_case("demo", "visited_list", seed=7)
    for i, rid in enumerate(["debug_log", "heuristic_recompute", "ef_default", "stack_gather"]):
        make_case(f"dev-{i}", rid, seed=50_000 + i)
    index = [load_meta(CASES / cid) for cid, _, _ in spec]
    (CASES / "index.json").write_text(json.dumps(index, indent=1))
    n_reg = sum(m["kind"] == "regression" for m in index)
    print(f"built {len(index)} cases ({n_reg} regression, {len(index) - n_reg} control) in {CASES}")


if __name__ == "__main__":
    main()
