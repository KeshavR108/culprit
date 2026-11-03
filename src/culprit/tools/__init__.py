"""Tool registry: name -> module, plus the JSON schemas the LLM sees."""

from pydantic import ValidationError

from ..sandbox import SandboxError
from ..session import BudgetExceeded

from . import (bisect_commits, profile, propose_patch, read_file, run_benchmark, search_code,
               show_diff, verify_patch)

TOOLS = {m.NAME: m for m in (run_benchmark, bisect_commits, profile, show_diff, read_file,
                             search_code, propose_patch, verify_patch)}


def _clean(schema: dict) -> dict:
    """Pydantic schemas carry titles and anyOf-null noise that confuse small models."""
    props = {}
    for name, p in schema.get("properties", {}).items():
        q = {k: v for k, v in p.items() if k not in ("title", "default")}
        if "anyOf" in q:
            q.update(next(t for t in q.pop("anyOf") if t.get("type") != "null"))
        props[name] = q
    return {"type": "object", "properties": props, "required": schema.get("required", [])}


def schemas() -> list[dict]:
    return [{"type": "function",
             "function": {"name": m.NAME, "description": m.DESCRIPTION,
                          "parameters": _clean(m.Args.model_json_schema())}}
            for m in TOOLS.values()]


def call(session, name: str, arguments: dict) -> dict:
    """Validate arguments and run a tool."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name!r}; available: {sorted(TOOLS)}"}
    mod = TOOLS[name]
    try:
        args = mod.Args.model_validate(arguments or {})
    except ValidationError as e:
        return {"error": f"invalid arguments for {name}: {e.errors(include_url=False)}"}
    try:
        return mod.run(session, args)
    except (ValueError, SandboxError, BudgetExceeded) as e:
        return {"error": str(e)}
