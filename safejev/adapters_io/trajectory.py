"""Generic trajectories: a list of steps ``{"role": ..., "content": ...}``.

Roles: ``user``, ``assistant``, ``tool_call`` (content = {"tool", "arguments"}), ``tool_result``.
``to_step_inputs`` cuts a trajectory at every tool call, giving one pre-execution input per
call: (goal, history so far, proposed call) (design doc §9.3).
"""
from __future__ import annotations

from typing import Any, Iterator

from ..core.detector import ScanInput

ACTION_ROLES = ("tool_call", "action")


def normalize_action(content: Any) -> dict[str, Any]:
    """Accept {"tool", "arguments"}, {"name", "arguments"}, {"tool", "tool_input"} or a string."""
    if isinstance(content, dict):
        tool = content.get("tool") or content.get("name") or content.get("function") or ""
        args = content.get("arguments", content.get("args", content.get("tool_input", {})))
        if isinstance(tool, dict):                     # OpenAI {"function": {"name", "arguments"}}
            args, tool = tool.get("arguments", args), tool.get("name", "")
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"_raw": args}
        return {"tool": str(tool), "arguments": args}
    return {"tool": str(content), "arguments": {}}


def to_step_inputs(steps: list[dict[str, Any]], user_goal: str | None = None
                   ) -> Iterator[tuple[int, ScanInput]]:
    """Yield (step_index, ScanInput) for each tool call. ``user_goal`` defaults to the first
    user message."""
    goal = user_goal
    if goal is None:
        goal = next((str(s.get("content", "")) for s in steps if s.get("role") == "user"), "")
    history: list[dict[str, Any]] = []
    for i, s in enumerate(steps):
        role = s.get("role")
        if role in ACTION_ROLES:
            action = normalize_action(s.get("content"))
            yield i, ScanInput(user_input=goal, agent_action=action, history=list(history),
                               tool_schema=s.get("tool_schema"))
            history.append({"role": "tool_call", "content": action})
        elif role == "user" and str(s.get("content", "")) == goal and not history:
            continue
        else:
            history.append({"role": role or "step", "content": s.get("content", "")})
