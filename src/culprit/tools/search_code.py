"""Tool: regex search over the repository's tracked text files at a commit."""

import re

from pydantic import BaseModel, Field

from .. import sandbox
from ..sandbox import safe_path

NAME = "search_code"
DESCRIPTION = "Search the repository's files for a regular expression (or plain text). Returns file:line matches."
MAX_HITS = 40


class Args(BaseModel):
    pattern: str = Field(description="regex or plain text to search for")
    commit: str | None = Field(None, description="commit hash (default: bad commit)")


def run(s, a: Args) -> dict:
    sha = s.resolve(a.commit) if a.commit else s.bad
    root = s.wt.get(sha)
    try:
        rx = re.compile(a.pattern)
    except re.error:
        rx = re.compile(re.escape(a.pattern))
    hits = []
    for rel in sandbox.git(root, "ls-files").split():
        if not rel.endswith((".py", ".md", ".toml", ".cfg", ".txt", ".pyx", ".c", ".h")):
            continue
        for i, line in enumerate(safe_path(root, rel).read_text(errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:160]}")
                if len(hits) >= MAX_HITS:
                    return {"matches": hits, "truncated": True}
    return {"matches": hits}
