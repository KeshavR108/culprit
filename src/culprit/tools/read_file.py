"""Tool: read a window of a file, with line numbers, at a given commit."""

from pydantic import BaseModel, Field

from ..sandbox import safe_path

NAME = "read_file"
DESCRIPTION = ("Read lines of a file in the repository, with line numbers. Default commit: the bad commit. "
               "Reads at most 150 lines per call.")
MAX_LINES = 150


class Args(BaseModel):
    path: str = Field(description="repo-relative path, as shown in show_diff or search_code")
    start_line: int = Field(1, description="first line to show (1-based)")
    end_line: int | None = Field(None, description="last line to show")
    commit: str | None = Field(None, description="commit hash (default: bad commit)")


def run(s, a: Args) -> dict:
    sha = s.resolve(a.commit) if a.commit else s.bad
    root = s.wt.get(sha)
    p = safe_path(root, a.path)
    if not p.is_file():
        return {"error": f"no such file at {sha[:10]}: {a.path}"}
    lines = p.read_text(errors="replace").splitlines()
    start = max(1, a.start_line)
    end = min(len(lines), a.end_line or start + MAX_LINES - 1, start + MAX_LINES - 1)
    body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(start, end + 1))
    return {"path": a.path, "commit": sha[:10], "lines": f"{start}-{end} of {len(lines)}", "content": body}
