"""Tool: validate a proposed fix."""

import difflib
import re

from pydantic import BaseModel, Field

from .. import sandbox
from ..sandbox import safe_path
from ..schemas import Patch

NAME = "propose_patch"
DESCRIPTION = ("Propose a fix against the bad commit. Preferred form: `file`, `old_code` (exact lines copied "
               "from read_file, without the line-number prefix) and `new_code` (the replacement). "
               "Alternatively pass a unified `diff`. The patch is validated, not applied.")
MAX_LINES_CHANGED = 60
_LINENO = re.compile(r"^\s*\d+\| ?")
TOLERANT = True


class Args(BaseModel):
    file: str | None = Field(None, description="repo-relative path of the file to change")
    old_code: str | None = Field(None, description="exact existing code to replace")
    new_code: str | None = Field(None, description="replacement code")
    diff: str | None = Field(None, description="alternatively, a unified diff")


def _strip_line_numbers(code: str) -> str:
    lines = code.splitlines()
    if lines and all(_LINENO.match(l) or not l.strip() for l in lines):
        return "\n".join(_LINENO.sub("", l, count=1) for l in lines)
    return code


def _clean(code: str, side: str) -> list[str]:
    """Remove copy artifacts."""
    lines = [_LINENO.sub("", l, count=1) for l in code.splitlines()]
    diff_like = any(l.startswith(("+", "-")) for l in lines) and not all(l.startswith(("+", "-")) for l in lines if l)
    if diff_like or all(l[:1] in "+- " for l in lines):
        out = []
        for l in lines:
            if l.startswith("-"):
                continue
            out.append(l[1:] if l[:1] in "+ " and diff_like else l)
        return out
    return lines


def _indent(l: str) -> int:
    return len(l) - len(l.lstrip(" "))


def _tolerant_locate(text: str, old: str, new: str):
    """Find old_code ignoring artifacts and indentation; return (old_exact, new_reindented) or None."""
    file_lines = text.splitlines(True)
    old_lines = _clean(old, "old")
    key = [l.strip() for l in old_lines if l.strip()]
    if not key:
        return None
    nb = [(i, l.strip()) for i, l in enumerate(file_lines) if l.strip()]
    hits = [j for j in range(len(nb) - len(key) + 1) if [s for _, s in nb[j:j + len(key)]] == key]
    if len(hits) != 1:
        return None
    start, end = nb[hits[0]][0], nb[hits[0] + len(key) - 1][0] + 1
    old_exact = "".join(file_lines[start:end])
    cleaned_new = _clean(new, "new")
    first_new = next((l for l in cleaned_new if l.strip()), "")
    delta = _indent(file_lines[start]) - _indent(first_new)
    new_lines = []
    for l in cleaned_new:
        if not l.strip():
            new_lines.append("")
        elif delta >= 0:
            new_lines.append(" " * delta + l)
        else:
            new_lines.append(l[min(-delta, _indent(l)):])
    new_text = "\n".join(new_lines)
    if old_exact.endswith("\n"):
        new_text += "\n"
    return old_exact, new_text


def edit_to_diff(root, file: str, old: str, new: str) -> str:
    p = safe_path(root, file)
    if not p.is_file():
        raise ValueError(f"no such file: {file}")
    text = p.read_text()
    old, new = _strip_line_numbers(old), _strip_line_numbers(new)
    n = text.count(old)
    if TOLERANT and n == 1 and not re.search(r"(?m)^" + re.escape(old), text):
        n = 0
    if n == 0:
        norm = lambda t: "\n".join(l.rstrip() for l in t.splitlines())
        if norm(text).count(norm(old)) == 1 and re.search(r"(?m)^" + re.escape(norm(old)), norm(text)):
            text, old, new = norm(text) + ("\n" if text.endswith("\n") else ""), norm(old), norm(new)
            n = 1
        elif TOLERANT and (found := _tolerant_locate(text, old, new)):
            old, new = found
            n = 1
        else:
            raise ValueError("old_code was not found in the file; copy it exactly from read_file "
                             "(without the line-number prefix)")
    if n > 1:
        raise ValueError(f"old_code appears {n} times; include more surrounding lines so it is unique")
    updated = text.replace(old, new, 1)
    if updated == text:
        raise ValueError("new_code is identical to old_code")
    return "".join(difflib.unified_diff(text.splitlines(True), updated.splitlines(True),
                                        f"a/{file}", f"b/{file}"))


def count_changed(diff: str) -> int:
    return sum(1 for l in diff.splitlines()
               if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---")))


def files_in(diff: str) -> list[str]:
    return sorted({l[6:].strip() for l in diff.splitlines() if l.startswith("+++ b/")})


def validate(s, diff: str) -> Patch:
    if not diff.endswith("\n"):
        diff += "\n"
    files = files_in(diff)
    if not files:
        raise ValueError("not a unified diff: no '+++ b/<file>' header found")
    scratch = s.wt.get(s.bad)
    for f in files:
        safe_path(scratch, f)
    n = count_changed(diff)
    if n > MAX_LINES_CHANGED:
        raise ValueError(f"patch changes {n} lines; the limit is {MAX_LINES_CHANGED}")
    patch_file = s.wt.root / f"check-{s.patch_hash(diff)}.diff"
    patch_file.write_text(diff)
    r = sandbox.run(["git", "apply", "--check", "--recount", str(patch_file)], cwd=scratch)
    if r.returncode != 0:
        raise ValueError(f"patch does not apply to the bad commit: {r.stderr.strip()[:400]}")
    return Patch(diff=diff, files=files, lines_changed=n)


def run(s, a: Args) -> dict:
    try:
        if a.diff and not a.old_code:
            diff = a.diff
        elif a.file and a.old_code is not None and a.new_code is not None:
            diff = edit_to_diff(s.wt.get(s.bad), a.file, a.old_code, a.new_code)
        else:
            return {"error": "give either `diff`, or `file` + `old_code` + `new_code`"}
        s.patch = validate(s, diff)
    except (ValueError, sandbox.SandboxError) as e:
        return {"valid": False, "error": str(e)}
    return {"valid": True, "files": s.patch.files, "lines_changed": s.patch.lines_changed,
            "diff": s.patch.diff, "next": "call verify_patch to measure it"}
