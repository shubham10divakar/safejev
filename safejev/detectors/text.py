"""Render a ScanInput into the (context, candidate) text pair a cross-encoder reads.

The same function is used at training time (from a Record) and at inference time, so the
model never sees two different formats. Long fields are truncated from the left for history
(keep the most recent steps) and from the right for everything else.
"""
from __future__ import annotations

import json
from typing import Any

from ..core.detector import ScanInput

MAX_FIELD_CHARS = 1500


def _clip(s: str, n: int = MAX_FIELD_CHARS) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def _dump(x: Any) -> str:
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(x)


def render_history(history: list[dict[str, Any]], last_k: int | None = None) -> str:
    steps = history if last_k is None else history[-last_k:] if last_k > 0 else []
    lines = [f"[{h.get('role', 'step')}] {_dump(h.get('content', ''))}" for h in steps]
    text = "\n".join(lines)
    return text if len(text) <= 2 * MAX_FIELD_CHARS else "..." + text[-2 * MAX_FIELD_CHARS:]


def render_pair(inp: ScanInput, granularity: str, last_k: int | None = None) -> tuple[str, str]:
    """(context, candidate). ``last_k`` limits history (experiment E4: 0, 1, 3, None = all)."""
    if granularity == "text":
        ctx = f"system: {_clip(inp.system_prompt)}" if inp.system_prompt else ""
        parts = []
        if inp.user_input:
            parts.append(f"user: {_clip(inp.user_input)}")
        for c in inp.untrusted_context:
            parts.append(f"untrusted: {_clip(c)}")
        if inp.response:
            parts.append(f"response: {_clip(inp.response)}")
        return ctx, "\n".join(parts)

    ctx_parts = [f"goal: {_clip(inp.user_input)}"]
    hist = render_history(inp.history, last_k)
    if hist:
        ctx_parts.append(f"history:\n{hist}")
    for c in inp.untrusted_context:
        ctx_parts.append(f"untrusted: {_clip(c)}")
    if inp.tool_schema:
        ctx_parts.append(f"tool schema: {_clip(_dump(inp.tool_schema))}")
    action = inp.agent_action or {}
    cand = f"call {action.get('tool', '?')}({_clip(_dump(action.get('arguments', {})))})"
    return "\n".join(ctx_parts), cand
