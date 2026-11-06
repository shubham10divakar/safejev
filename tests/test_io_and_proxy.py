import io
import json
import sys
from pathlib import Path

from safejev import SafeJEV
from safejev.adapters_io import langchain, mcp, openai, trajectory
from safejev.governance import Policy
from safejev.middleware import MCPProxy, wrap_mcp_config

HERE = Path(__file__).parent


def test_openai_pending_calls():
    msgs = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "Find duplicate files in /documents"},
        {"role": "assistant", "tool_calls": [{"id": "1", "type": "function",
                                              "function": {"name": "list_files", "arguments": '{"path": "/documents"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "a.pdf b.pdf"},
        {"role": "assistant", "tool_calls": [
            {"id": "2", "type": "function", "function": {"name": "delete_file", "arguments": '{"path": "/documents/b.pdf"}'}},
            {"id": "3", "type": "function", "function": {"name": "x", "arguments": "not json"}}]},
    ]
    tools = [{"type": "function", "function": {"name": "delete_file", "parameters": {"type": "object"},
                                               "x-safejev-sensitive": True}}]
    ins = openai.pending_tool_calls(msgs, tools)
    assert len(ins) == 2
    a = ins[0]
    assert a.user_input.startswith("Find") and a.system_prompt == "be helpful"
    assert a.agent_action == {"tool": "delete_file", "arguments": {"path": "/documents/b.pdf"}}
    assert [h["role"] for h in a.history] == ["tool_call", "tool_result"]
    assert a.untrusted_context == ["a.pdf b.pdf"] and a.tool_schema["sensitive"]
    assert ins[1].agent_action["arguments"] == {"_raw": "not json"}
    assert openai.pending_tool_calls(msgs[:2]) == []
    assert openai.from_chat(msgs).untrusted_context == ["a.pdf b.pdf"]


def test_trajectory_steps():
    steps = [{"role": "user", "content": "clean /tmp"},
             {"role": "tool_call", "content": {"name": "ls", "args": {"p": "/tmp"}}},
             {"role": "tool_result", "content": "a b"},
             {"role": "tool_call", "content": {"tool": "rm", "arguments": '{"p": "/tmp/a"}'}}]
    out = list(trajectory.to_step_inputs(steps))
    assert [i for i, _ in out] == [1, 3]
    assert out[1][1].agent_action == {"tool": "rm", "arguments": {"p": "/tmp/a"}}
    assert len(out[1][1].history) == 2 and out[0][1].history == []
    assert trajectory.normalize_action("noop") == {"tool": "noop", "arguments": {}}


def test_langchain_steps_without_langchain():
    class A:
        tool, tool_input = "search", "cats"
    inp = langchain.from_intermediate_steps("find cats", [(A(), "result")],
                                            {"tool": "send", "tool_input": '{"to": "x"}'})
    assert inp.agent_action == {"tool": "send", "arguments": {"to": "x"}}
    assert inp.history[0]["content"] == {"tool": "search", "arguments": {"input": "cats"}}


def test_mcp_helpers():
    sch = mcp.schemas_from_list({"tools": [
        {"name": "d", "annotations": {"destructiveHint": True}},
        {"name": "w", "annotations": {"readOnlyHint": False, "openWorldHint": True}},
        {"name": "r"}]})
    assert sch["d"]["sensitive"] and sch["w"]["sensitive"] and not sch["r"]["sensitive"]
    b = mcp.blocked_result(7, "d", "block", "authz", ["nope"], "hint")
    assert b["id"] == 7 and b["result"]["isError"] and "nope" in b["result"]["content"][0]["text"]


def _rpc(i, method, **params):
    return {"jsonrpc": "2.0", "id": i, "method": method, "params": params}


def test_proxy_handlers_block_destructive():
    sj = SafeJEV(modules=["agent"], backend="heuristic", policy=Policy(require_approval_tools=[]))
    px = MCPProxy(sj, user_goal="Find duplicate files in /documents")
    fwd, reply = px.on_client_message(_rpc(1, "tools/list"))
    assert fwd and reply is None
    px.on_server_message({"jsonrpc": "2.0", "id": 1, "result": {"tools": [
        {"name": "delete_file", "annotations": {"destructiveHint": True}}]}})
    assert px.schemas["delete_file"]["sensitive"]
    fwd, reply = px.on_client_message(_rpc(2, "tools/call", name="list_files", arguments={"path": "/documents"}))
    assert fwd is not None and reply is None
    px.on_server_message({"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "a b"}]}})
    assert px.history[-1] == {"role": "tool_result", "content": "a b"}
    fwd, reply = px.on_client_message(_rpc(3, "tools/call", name="delete_file",
                                           arguments={"path": "/documents/report.pdf"}))
    assert fwd is None and reply["result"]["isError"] and "approval denied" in reply["result"]["content"][0]["text"]
    assert px.on_client_message({"jsonrpc": "2.0", "method": "notifications/initialized"})[0]


def test_proxy_end_to_end_subprocess():
    sj = SafeJEV(modules=["agent"], backend="heuristic")
    px = MCPProxy(sj, [sys.executable, str(HERE / "fake_mcp_server.py")],
                  user_goal="Read /data/a.txt")
    lines = [_rpc(1, "tools/list"),
             _rpc(2, "tools/call", name="read_file", arguments={"path": "/data/a.txt"}),
             _rpc(3, "tools/call", name="delete_file", arguments={"path": "/data/a.txt"})]
    cin = io.StringIO("".join(json.dumps(m) + "\n" for m in lines))
    cout = io.StringIO()
    assert px.run(cin, cout) == 0
    got = {m["id"]: m for m in map(json.loads, cout.getvalue().splitlines())}
    assert set(got) == {1, 2, 3}
    assert "ran read_file" in got[2]["result"]["content"][0]["text"]
    assert got[3]["result"]["isError"]          # destructive + not requested: never reached server


def test_wrap_config():
    cfg = {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "server-fs", "/x"]},
                          "remote": {"url": "https://x"}}}
    w = wrap_mcp_config(cfg, policy_path="p.yaml")
    assert w["mcpServers"]["fs"] == {"command": "safejev",
                                     "args": ["proxy", "--policy", "p.yaml", "--", "npx", "-y", "server-fs", "/x"]}
    assert w["mcpServers"]["remote"] == {"url": "https://x"} and cfg["mcpServers"]["fs"]["command"] == "npx"
    assert wrap_mcp_config(w) == w
