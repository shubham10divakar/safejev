"""LangChain integration without a hard dependency.

- ``from_intermediate_steps``: (AgentAction, observation) pairs → ScanInput for the next action.
- ``callback_handler(sj, user_goal)``: a ``BaseCallbackHandler`` whose ``on_tool_start`` raises
  :class:`safejev.ToolBlocked` before a blocked tool runs (needs ``langchain-core``).
"""
from __future__ import annotations

import json
from typing import Any

from ..core.detector import ScanInput


def _action(a: Any) -> dict:
    tool = getattr(a, "tool", None) or (a.get("tool") if isinstance(a, dict) else "")
    ti = getattr(a, "tool_input", None)
    if ti is None and isinstance(a, dict):
        ti = a.get("tool_input", {})
    if isinstance(ti, str):
        try:
            ti = json.loads(ti)
        except json.JSONDecodeError:
            ti = {"input": ti}
    return {"tool": tool or "", "arguments": ti if isinstance(ti, dict) else {"input": ti}}


def from_intermediate_steps(user_goal: str, steps: list[tuple[Any, Any]], next_action: Any) -> ScanInput:
    history: list[dict] = []
    untrusted = []
    for act, obs in steps:
        history.append({"role": "tool_call", "content": _action(act)})
        history.append({"role": "tool_result", "content": str(obs)})
        untrusted.append(str(obs))
    return ScanInput(user_input=user_goal, agent_action=_action(next_action), history=history,
                     untrusted_context=untrusted)


def callback_handler(sj, user_goal: str = "", sensitive_tools: tuple[str, ...] = ()):
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("callback_handler needs langchain-core: pip install langchain-core") from e
    from ..engine import ToolBlocked

    class SafeJEVCallbackHandler(BaseCallbackHandler):
        raise_error = True        # make LangChain propagate ToolBlocked

        def __init__(self):
            self.history: list[dict] = []

        def on_tool_start(self, serialized, input_str, *, inputs=None, **kw):
            tool = (serialized or {}).get("name", "")
            args = inputs if isinstance(inputs, dict) else _action({"tool": tool, "tool_input": input_str})["arguments"]
            v = sj.check_tool_call(tool, args, user_goal=user_goal, history=self.history,
                                   sensitive=tool in sensitive_tools)
            if not v.allowed:
                raise ToolBlocked(tool, v.result, v.approval)
            self.history.append({"role": "tool_call", "content": {"tool": tool, "arguments": args}})

        def on_tool_end(self, output, **kw):
            self.history.append({"role": "tool_result", "content": str(output)[:2000]})

    return SafeJEVCallbackHandler()
