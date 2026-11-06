"""OpenAI-style chat messages and tool definitions → ScanInput."""
from __future__ import annotations

import json
from typing import Any

from ..core.detector import ScanInput


def _text(content: Any) -> str:
    if isinstance(content, list):     # content parts
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def tool_schemas(tools: list[dict] | None) -> dict[str, dict]:
    """{"type": "function", "function": {"name", "parameters", ...}} → {name: schema}."""
    out = {}
    for t in tools or []:
        f = t.get("function", t)
        out[f["name"]] = {"name": f["name"], "params": f.get("parameters", {}),
                          "sensitive": bool(f.get("x-safejev-sensitive", False))}
    return out


def pending_tool_calls(messages: list[dict], tools: list[dict] | None = None) -> list[ScanInput]:
    """One ScanInput per tool call in the **last** assistant message, with everything before it
    as history. Tool messages are untrusted content."""
    schemas = tool_schemas(tools)
    goal = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    system = next((_text(m.get("content")) for m in messages if m.get("role") == "system"), "")
    last = max((i for i, m in enumerate(messages)
                if m.get("role") == "assistant" and m.get("tool_calls")), default=None)
    if last is None:
        return []
    history, untrusted = [], []
    first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    for i, m in enumerate(messages[:last]):
        role = m.get("role")
        if i == first_user:
            continue                       # that is the goal, not history
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                history.append({"role": "tool_call", "content": _call(tc)})
        elif role == "tool":
            history.append({"role": "tool_result", "content": _text(m.get("content"))})
            untrusted.append(_text(m.get("content")))
        elif role in ("user", "assistant"):
            history.append({"role": role, "content": _text(m.get("content"))})
    out = []
    for tc in messages[last]["tool_calls"]:
        call = _call(tc)
        out.append(ScanInput(user_input=goal, agent_action=call, history=list(history),
                             untrusted_context=list(untrusted), system_prompt=system,
                             tool_schema=schemas.get(call["tool"])))
    return out


def _call(tc: dict) -> dict:
    f = tc.get("function", tc)
    args = f.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {"_raw": args}
    return {"tool": f.get("name", ""), "arguments": args if isinstance(args, dict) else {"_raw": args}}


def from_chat(messages: list[dict]) -> ScanInput:
    """Text scan of a chat: the last user message, with tool outputs as untrusted context."""
    users = [m for m in messages if m.get("role") == "user"]
    return ScanInput(
        user_input=_text(users[-1].get("content")) if users else "",
        untrusted_context=[_text(m.get("content")) for m in messages if m.get("role") == "tool"],
        system_prompt=next((_text(m.get("content")) for m in messages if m.get("role") == "system"), ""))
