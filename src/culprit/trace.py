"""Every step of an investigation is appended to a JSONL file, one event per line."""

from __future__ import annotations

import json
import time
from pathlib import Path


class Trace:
    def __init__(self, path: Path | None, echo=None):
        self.path = Path(path) if path else None
        self.echo = echo
        self.t0 = time.time()
        self.events: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def log(self, kind: str, **data) -> None:
        event = {"t": round(time.time() - self.t0, 3), "kind": kind, **data}
        self.events.append(event)
        if self.echo:
            self.echo(event)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")
