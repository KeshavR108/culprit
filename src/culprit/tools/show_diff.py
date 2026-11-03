"""Tool: show what one commit changed (size-limited)."""

from pydantic import BaseModel, Field

from .. import sandbox

NAME = "show_diff"
DESCRIPTION = "Show the diff of one commit (its message, changed files and the patch, truncated if large)."
MAX_CHARS = 6000


class Args(BaseModel):
    commit: str = Field(description="commit hash")


def run(s, a: Args) -> dict:
    sha = s.resolve(a.commit)
    text = sandbox.git(s.repo, "show", "--format=commit %H%nSubject: %s%n", "--stat", "--patch",
                       "--unified=3", sha)
    if len(text) <= MAX_CHARS:
        return {"diff": text}
    stat = sandbox.git(s.repo, "show", "--format=commit %H%nSubject: %s%n", "--stat", sha)
    return {"diff": stat + "\n" + text[len(stat):][:MAX_CHARS - len(stat)] + "\n... [truncated]",
            "truncated": True}
