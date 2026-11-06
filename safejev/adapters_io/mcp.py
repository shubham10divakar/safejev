"""MCP (Model Context Protocol) JSON-RPC messages → actions and tool schemas."""
from __future__ import annotations

from typing import Any


def is_tool_call(msg: dict) -> bool:
    return msg.get("method") == "tools/call" and "id" in msg


def action_from_call(msg: dict) -> dict[str, Any]:
    p = msg.get("params") or {}
    args = p.get("arguments") or {}
    return {"tool": str(p.get("name", "")), "arguments": args if isinstance(args, dict) else {"_raw": args}}


def schemas_from_list(result: dict) -> dict[str, dict]:
    """``tools/list`` result → {name: schema}. MCP annotations mark destructive / open-world
    tools; those count as sensitive."""
    out = {}
    for t in (result or {}).get("tools", []):
        ann = t.get("annotations") or {}
        sensitive = bool(ann.get("destructiveHint")) or (
            ann.get("readOnlyHint") is False and bool(ann.get("openWorldHint")))
        out[t["name"]] = {"name": t["name"], "params": t.get("inputSchema") or {},
                          "sensitive": sensitive}
    return out


def result_text(result: dict) -> str:
    """Concatenate text content of a ``tools/call`` result."""
    parts = (result or {}).get("content") or []
    return "\n".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")


def blocked_result(req_id: Any, tool: str, decision: str, rule: str, reasons: list[str],
                   hint: str = "") -> dict:
    """A tools/call *result* (not a JSON-RPC error), so the model sees why and can adapt."""
    text = f"SafeJEV {decision}: the call to {tool!r} was not executed (rule: {rule})."
    if reasons:
        text += " Reasons: " + "; ".join(reasons[:5])
    if hint:
        text += f" Hint: {hint}"
    return {"jsonrpc": "2.0", "id": req_id,
            "result": {"content": [{"type": "text", "text": text}], "isError": True}}
