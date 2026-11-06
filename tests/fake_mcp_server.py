"""Minimal stdio MCP server for proxy tests: tools/list + tools/call that echoes."""
import json
import sys

TOOLS = [
    {"name": "read_file", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
     "annotations": {"readOnlyHint": True}},
    {"name": "delete_file", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
     "annotations": {"destructiveHint": True}},
]

for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    if msg["method"] == "tools/list":
        res = {"tools": TOOLS}
    elif msg["method"] == "tools/call":
        p = msg["params"]
        res = {"content": [{"type": "text", "text": f"ran {p['name']} {json.dumps(p['arguments'])}"}]}
    else:
        res = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": res}) + "\n")
    sys.stdout.flush()
