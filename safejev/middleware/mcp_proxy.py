"""MCP screening proxy (design doc §4): sits between an MCP client and one stdio MCP server and
screens every ``tools/call`` before it reaches the server.

    client  <-- stdio -->  safejev proxy  <-- stdio -->  real server (subprocess)

- ``tools/list`` responses are read to learn tool schemas and MCP annotations
  (``destructiveHint`` etc. count as sensitive).
- ``tools/call`` requests go through ``SafeJEV.check_tool_call``. A blocked call is answered by
  the proxy with an ``isError`` tool result that explains why; it never reaches the server.
- Tool results are kept as history and untrusted context for later checks (injected goals).
- If the check itself fails, sensitive tools are blocked (fail-closed); others are forwarded
  only when ``fail_closed`` is off in the policy.

MCP's stdio transport is newline-delimited JSON-RPC, so the proxy works line by line.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any, TextIO

from ..adapters_io import mcp


class MCPProxy:
    def __init__(self, sj, server_cmd: list[str] | None = None, *, user_goal: str = "",
                 sensitive_tools: tuple[str, ...] = (), env: dict | None = None,
                 max_history: int = 40):
        self.sj = sj
        self.server_cmd = server_cmd
        self.env = env
        self.user_goal = user_goal
        self.sensitive_tools = set(sensitive_tools)
        self.schemas: dict[str, dict] = {}
        self.history: list[dict] = []
        self.max_history = max_history
        self._pending: dict[Any, str] = {}          # request id -> method / tool
        self._lock = threading.Lock()

    # ---- pure message handlers (unit-testable) ---------------------------------------------
    def on_client_message(self, msg: dict) -> tuple[dict | None, dict | None]:
        """Returns (forward_to_server, reply_to_client)."""
        if "id" in msg and "method" in msg:
            self._pending[msg["id"]] = msg["method"]
        if not mcp.is_tool_call(msg):
            return msg, None
        action = mcp.action_from_call(msg)
        tool = action["tool"]
        schema = self.schemas.get(tool)
        sensitive = tool in self.sensitive_tools or bool((schema or {}).get("sensitive"))
        try:
            v = self.sj.check_tool_call(tool, action["arguments"], user_goal=self.user_goal,
                                        history=list(self.history), tool_schema=schema,
                                        untrusted_context=[h["content"] for h in self.history
                                                           if h["role"] == "tool_result"][-5:],
                                        sensitive=sensitive)
        except Exception as e:  # noqa: BLE001 - fail closed unless the policy says otherwise
            if self.sj.policy.fail_closed:
                self._pending.pop(msg["id"], None)
                return None, mcp.blocked_result(msg["id"], tool, "block", "fail_closed",
                                                [f"{type(e).__name__}: {e}"])
            return msg, None
        if not v.allowed:
            self._pending.pop(msg["id"], None)
            reasons = list(v.result.reasons)
            if v.approval is not None:
                reasons.insert(0, f"approval denied by {v.approval.approver}"
                                  + (f" ({v.approval.note})" if v.approval.note else ""))
            return None, mcp.blocked_result(msg["id"], tool, v.result.decision,
                                            v.result.policy_rule, reasons, v.result.repair_hint)
        self._pending[msg["id"]] = f"tools/call:{tool}"
        self._remember({"role": "tool_call", "content": action})
        return msg, None

    def on_server_message(self, msg: dict) -> dict:
        """Observe a server → client message and return it unchanged."""
        method = self._pending.pop(msg.get("id"), None) if "id" in msg and "method" not in msg else None
        if method == "tools/list" and isinstance(msg.get("result"), dict):
            self.schemas.update(mcp.schemas_from_list(msg["result"]))
        elif method and method.startswith("tools/call:"):
            text = mcp.result_text(msg.get("result") or {}) or json.dumps(msg.get("error", ""))
            self._remember({"role": "tool_result", "content": text[:2000]})
        return msg

    def _remember(self, step: dict):
        with self._lock:
            self.history.append(step)
            del self.history[:-self.max_history]

    # ---- stdio loop ------------------------------------------------------------------------
    def run(self, client_in: TextIO | None = None, client_out: TextIO | None = None) -> int:
        if not self.server_cmd:
            raise ValueError("server_cmd is required to run the proxy")
        client_in = client_in or sys.stdin
        client_out = client_out or sys.stdout
        proc = subprocess.Popen(self.server_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                text=True, encoding="utf-8", bufsize=1, env=self.env)
        out_lock = threading.Lock()

        def to_client(obj_or_line):
            line = obj_or_line if isinstance(obj_or_line, str) else json.dumps(obj_or_line)
            with out_lock:
                client_out.write(line.rstrip("\n") + "\n")
                client_out.flush()

        def pump_server():
            for line in proc.stdout:
                if not line.strip():
                    continue
                try:
                    self.on_server_message(json.loads(line))
                except (json.JSONDecodeError, AttributeError):
                    pass                     # not JSON-RPC: pass through untouched
                to_client(line)

        t = threading.Thread(target=pump_server, daemon=True)
        t.start()
        try:
            for line in client_in:
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    proc.stdin.write(line)
                    proc.stdin.flush()
                    continue
                fwd, reply = self.on_client_message(msg) if isinstance(msg, dict) else (msg, None)
                if reply is not None:
                    to_client(reply)
                if fwd is not None:
                    proc.stdin.write(json.dumps(fwd) + "\n")
                    proc.stdin.flush()
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
            t.join(timeout=5)
            try:
                rc = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
        return rc


def wrap_mcp_config(config: dict, policy_path: str | None = None, proxy_cmd: str = "safejev") -> dict:
    """Rewrite an ``{"mcpServers": {name: {command, args, env}}}`` config so that every stdio
    server runs behind ``safejev proxy``."""
    out = json.loads(json.dumps(config))
    for name, srv in (out.get("mcpServers") or {}).items():
        if "command" not in srv or srv.get("command") == proxy_cmd:
            continue
        args = ["proxy"] + (["--policy", policy_path] if policy_path else []) + \
            ["--", srv["command"], *srv.get("args", [])]
        srv["command"], srv["args"] = proxy_cmd, args
    return out
