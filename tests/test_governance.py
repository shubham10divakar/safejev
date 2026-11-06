import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from safejev.core.detector import ScanInput
from safejev.core.fusion import fuse
from safejev.core.result import Detection, ScanResult
from safejev.governance import (ApprovalRequest, ArgConstraint, AuditLog, Authorizer,
                                CallbackApprover, CLIApprover, Policy, PolicyRule, RateLimit,
                                ToolRule, WebhookApprover, format_markdown, read_audit, summarize)

ROOT = Path(__file__).resolve().parents[1]


def det(t, sev, score=0.9, module="agent"):
    return Detection(module, t, score, "safejev/x@0", sev)


def call(tool, **args):
    return ScanInput(user_input="u", agent_action={"tool": tool, "arguments": args})


def test_severity_thresholds():
    p = Policy()
    assert p.decide(fuse([det("malformed_call", "low")]), call("t")).decision == "allow"
    d = p.decide(fuse([det("intent_mismatch", "high")]), call("t"))
    assert (d.decision, d.rule) == ("ask", "ask_on:high")
    assert p.decide(fuse([det("data_exfiltration", "critical")]), call("t")).decision == "block"


def test_rules_tools_and_failclosed():
    p = Policy(rules=[PolicyRule("block", type="injected_goal")],
               require_approval_tools=["delete_*"], sensitive_tools=["run_shell"])
    d = p.decide(fuse([det("injected_goal", "medium")]), call("search"))
    assert d.decision == "block" and "injected_goal" in d.rule
    assert p.decide(fuse([]), call("delete_file")).decision == "ask"
    d = p.decide(fuse([]), call("run_shell"), errors=["boom"])
    assert (d.decision, d.rule) == ("block", "fail_closed")
    assert p.decide(fuse([]), call("search"), errors=["boom"]).decision == "allow"
    assert p.decide(fuse([]), call("search"), errors=["boom"], sensitive=True).decision == "block"
    assert Policy(fail_closed=False).decide(fuse([]), call("x"), ["e"], True).decision == "allow"
    with pytest.raises(ValueError):
        PolicyRule("maybe")
    with pytest.raises(ValueError):
        Policy(ask_on=["huge"])


def test_authz_constraints_and_allowlist():
    a = Authorizer([ToolRule("read_file", args={"path": ArgConstraint(allowed_prefixes=["/data"])}),
                    ToolRule("http_get", args={"url": ArgConstraint(allowed_domains=["example.com"])}),
                    ToolRule("pay", args={"amount": ArgConstraint(max=100, required=True)}),
                    ToolRule("rm", allow=False)], default_allow=False)
    assert a.check("read_file", {"path": "/data/x"}) == []
    assert a.check("read_file", {"path": "/etc/passwd"})
    assert a.check("http_get", {"url": "https://api.example.com/v1"}) == []
    assert a.check("http_get", {"url": "https://example.com.evil.io/"})
    assert a.check("pay", {"amount": 500}) and a.check("pay", {})
    assert a.check("rm", {}) and a.check("unknown", {})
    e = ArgConstraint(allowed_domains=["corp.com"]).check("to", True, "bob@corp.com")
    assert e == []


def test_rate_limit_with_clock():
    t = [0.0]
    a = Authorizer([ToolRule("send", rate_limit=RateLimit.parse("2/60s"))], clock=lambda: t[0])
    assert a.check("send", {}) == [] and a.check("send", {}) == []
    assert a.check("send", {})
    t[0] = 61
    assert a.check("send", {}) == []
    assert RateLimit.parse("5/m").per_seconds == 60 and RateLimit.parse("3/h").calls == 3
    with pytest.raises(ValueError):
        RateLimit.parse("lots")


def test_policy_yaml_loads():
    p = Policy.from_yaml(ROOT / "configs" / "default_policy.yaml")
    assert isinstance(p.authz, Authorizer) and p.rules[0].type == "injected_goal"
    d = p.decide(fuse([]), call("transfer_funds", amount=5000))
    assert d.decision == "block" and d.rule == "authz"
    assert p.decide(fuse([]), call("read_file", path="/etc/shadow")).decision == "block"
    with pytest.raises(ValueError):
        Policy.from_dict({"blok_on": ["high"]})


def test_approvers_timeout_and_errors():
    req = ApprovalRequest("delete_file", {"path": "/x"}, "high", ["r"])
    assert CallbackApprover(lambda r: True)(req).approved
    slow = CallbackApprover(lambda r: time.sleep(2) or True, timeout=0.1)(req)
    assert not slow.approved and "timed out" in slow.note
    boom = CallbackApprover(lambda r: 1 / 0)(req)
    assert not boom.approved and "error" in boom.note
    out = io.StringIO()
    assert CLIApprover(stdin=io.StringIO("y\n"), stdout=out)(req).approved
    assert not CLIApprover(stdin=io.StringIO("sure\n"), stdout=out)(req).approved
    assert "delete_file" in out.getvalue()


def test_webhook_approver():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"approved": body["tool"] == "ok_tool"}).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/"
    try:
        assert WebhookApprover(url, timeout=5)(ApprovalRequest("ok_tool", {}, "high")).approved
        assert not WebhookApprover(url, timeout=5)(ApprovalRequest("bad", {}, "high")).approved
        assert not WebhookApprover("http://127.0.0.1:1/", timeout=1)(
            ApprovalRequest("ok_tool", {}, "high")).approved
    finally:
        srv.shutdown()


def test_audit_and_report(tmp_path):
    from safejev.governance import ApprovalOutcome
    log = AuditLog(tmp_path / "a" / "audit.jsonl")
    inp = call("delete_file", path="/x")
    r = ScanResult("ask", "high", [det("intent_mismatch", "high")], 12.0, "ask_on:high",
                   ["safejev/agent-precheck@0.1.0"])
    log.record(inp, r, ApprovalOutcome(True, "cli"))
    log.record(call("search"), ScanResult("allow", "none", [], 3.0, "default"))
    es = list(read_audit(log.path))
    assert len(es) == 2 and es[0]["input_hash"].startswith("sha256:") and "input" not in es[0]
    assert es[0]["input_hash"] == AuditLog(tmp_path / "b.jsonl").record(inp, r)["input_hash"]
    s = summarize(es)
    assert s["total"] == 2 and s["approval_frequency"] == 0.5 and s["approved"] == 1
    assert s["by_type"] == {"intent_mismatch": 1} and s["fpr_estimate_from_overrides"] == 1.0
    md = format_markdown(s)
    assert "approval frequency: 50.0%" in md and "| intent_mismatch | 1 |" in md
