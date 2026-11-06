"""Command line (design doc §8).

    safejev scan --text "Ignore previous instructions" --modules guard
    safejev scan --input prompt.txt --action '{"tool": "delete_file", "arguments": {"path": "/x"}}'
    safejev proxy --policy configs/default_policy.yaml -- npx -y @modelcontextprotocol/server-filesystem /tmp
    safejev proxy --mcp-config servers.json --server fs
    safejev proxy --wrap-config servers.json > servers.safejev.json
    safejev eval --adapter agent-precheck --data data/test.jsonl --out report.md
    safejev report --audit safejev_audit.jsonl
    safejev adapters
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _policy(path: str | None, audit: str | None = None):
    from .governance import Policy
    p = Policy.from_yaml(path) if path else Policy()
    if audit is not None:
        p.audit_log = audit or None
    return p


def _engine(a, policy, approver=None):
    from .engine import SafeJEV
    adapters = a.adapter if getattr(a, "adapter", None) else None
    modules = [m.strip() for m in (a.modules or "guard,agent").split(",") if m.strip()]
    return SafeJEV(modules=modules, adapters=adapters, policy=policy, backend=a.backend,
                   device=getattr(a, "device", None), approver=approver)


def cmd_scan(a) -> int:
    text = a.text if a.text is not None else (Path(a.input).read_text(encoding="utf-8") if a.input else "")
    if not text and not a.action and not sys.stdin.isatty() and not a.input:
        text = sys.stdin.read()
    action = json.loads(a.action) if a.action else None
    ctx = [Path(c).read_text(encoding="utf-8") for c in (a.context or [])]
    sj = _engine(a, _policy(a.policy, audit=a.audit))
    r = sj.scan(user_input=text, agent_action=action, untrusted_context=ctx,
                response=Path(a.response).read_text(encoding="utf-8") if a.response else "")
    if a.json:
        print(r.to_json(indent=2))
    else:
        print(f"decision: {r.decision}   risk: {r.risk_level}   rule: {r.policy_rule}   "
              f"({r.latency_ms:.1f} ms)")
        for d in r.detections:
            print(f"  [{d.severity:8}] {d.module}.{d.type}  score={d.score:.2f}  {d.adapter}")
            if d.reason:
                print(f"             {d.reason}")
        if r.repair_hint:
            print(f"hint: {r.repair_hint}")
        for e in r.errors:
            print(f"error: {e}", file=sys.stderr)
    return {"allow": 0, "ask": 2, "block": 3}[r.decision]


def cmd_proxy(a) -> int:
    from .governance import WebhookApprover
    from .middleware import MCPProxy, wrap_mcp_config
    if a.wrap_config:
        cfg = json.loads(Path(a.wrap_config).read_text(encoding="utf-8"))
        print(json.dumps(wrap_mcp_config(cfg, a.policy), indent=2))
        return 0
    cmd, env = list(a.command or []), None
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if a.mcp_config:
        cfg = json.loads(Path(a.mcp_config).read_text(encoding="utf-8"))
        servers = cfg.get("mcpServers", {})
        name = a.server or (next(iter(servers)) if len(servers) == 1 else None)
        if name not in servers:
            print(f"choose --server from: {', '.join(servers)}", file=sys.stderr)
            return 1
        srv = servers[name]
        cmd = [srv["command"], *srv.get("args", [])]
        if srv.get("env"):
            import os
            env = {**os.environ, **srv["env"]}
    if not cmd:
        print("give the server command after --, or --mcp-config", file=sys.stderr)
        return 1
    # stdin/stdout carry MCP traffic, so approval cannot prompt on the terminal
    approver = WebhookApprover(a.approve_webhook) if a.approve_webhook else None
    sj = _engine(a, _policy(a.policy, audit=a.audit), approver)
    px = MCPProxy(sj, cmd, user_goal=a.goal or "", env=env,
                  sensitive_tools=tuple(t for t in (a.sensitive or "").split(",") if t))
    return px.run()


def cmd_eval(a) -> int:
    from .core.schema import read_jsonl
    from .evaluation import predict, report, to_markdown
    sj = _engine(a, _policy(a.policy, audit=""))
    recs = list(read_jsonl(a.data))
    rep = report(predict(sj, recs), n_boot=a.boot)
    rep["adapters"] = sj.adapters
    rep["data"] = a.data
    md = to_markdown(rep, f"SafeJEV eval: {Path(a.data).name}")
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md if not a.json else json.dumps(rep, indent=2, default=str))
    return 0


def cmd_report(a) -> int:
    from .governance import format_markdown, read_audit, summarize
    s = summarize(read_audit(a.audit))
    print(json.dumps(s, indent=2) if a.json else format_markdown(s))
    return 0


def cmd_adapters(a) -> int:
    from .core.registry import list_adapters
    from .detectors import HEURISTICS
    for info in list_adapters(a.tier):
        impl = "heuristic+encoder" if info.name in HEURISTICS else "encoder only"
        print(f"{info.name:28} {info.module:7} {info.tier:9} {impl:18} {info.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    ap = argparse.ArgumentParser(prog="safejev", description="AI security and governance toolkit")
    ap.add_argument("--version", action="version", version=f"safejev {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, audit=True):
        p.add_argument("--modules", default=None, help="comma list: guard,misuse,agent")
        p.add_argument("--adapter", action="append", help="adapter name, hub id or local dir (repeat)")
        p.add_argument("--backend", default="auto", choices=["auto", "heuristic", "encoder"])
        p.add_argument("--policy", help="policy YAML")
        p.add_argument("--device")
        if audit:
            p.add_argument("--audit", default=None, help="audit log path ('' disables)")

    p = sub.add_parser("scan", help="scan a prompt / response / tool call")
    common(p)
    p.add_argument("--input", help="file with the user input")
    p.add_argument("--text", help="user input as a string")
    p.add_argument("--response", help="file with a model response to scan")
    p.add_argument("--action", help='tool call JSON: {"tool": ..., "arguments": {...}}')
    p.add_argument("--context", nargs="*", help="files with untrusted context (retrieved docs)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("proxy", help="MCP stdio screening proxy")
    common(p)
    p.add_argument("--mcp-config", help='{"mcpServers": {...}} file')
    p.add_argument("--server", help="server name in --mcp-config")
    p.add_argument("--wrap-config", help="print the config with every server behind the proxy")
    p.add_argument("--goal", help="the user goal, if known")
    p.add_argument("--sensitive", help="comma list of sensitive tools")
    p.add_argument("--approve-webhook", help="URL that answers approval requests")
    p.add_argument("command", nargs=argparse.REMAINDER, help="-- server command and args")
    p.set_defaults(fn=cmd_proxy)

    p = sub.add_parser("eval", help="evaluate adapters on a Record JSONL")
    common(p, audit=False)
    p.add_argument("--data", required=True)
    p.add_argument("--out", help="write report .md and .json here")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("report", help="summarise an audit log")
    p.add_argument("--audit", default="safejev_audit.jsonl")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("adapters", help="list known adapters")
    p.add_argument("--tier", choices=["user", "research"])
    p.set_defaults(fn=cmd_adapters)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
