"""Human approval (design doc §12). Every approver turns a timeout or an error into **deny**."""
from __future__ import annotations

import json
import sys
import threading
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, TextIO


@dataclass
class ApprovalRequest:
    tool: str
    arguments: dict[str, Any]
    risk_level: str
    reasons: list[str] = field(default_factory=list)
    user_input: str = ""


@dataclass
class ApprovalOutcome:
    approved: bool
    approver: str
    note: str = ""


class Approver(Protocol):
    name: str

    def __call__(self, req: ApprovalRequest) -> ApprovalOutcome: ...


class AutoDeny:
    name = "auto-deny"

    def __call__(self, req):
        return ApprovalOutcome(False, self.name, "no approver configured")


class AutoApprove:
    """For tests and dry runs only."""
    name = "auto-approve"

    def __call__(self, req):
        return ApprovalOutcome(True, self.name)


def _with_timeout(fn: Callable[[], bool], timeout: float | None) -> tuple[bool, str]:
    box: dict[str, Any] = {}

    def run():
        try:
            box["ok"] = bool(fn())
        except Exception as e:  # noqa: BLE001 - any approver error means deny
            box["err"] = repr(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return False, f"timed out after {timeout}s"
    if "err" in box:
        return False, f"approver error: {box['err']}"
    return box.get("ok", False), ""


class CallbackApprover:
    """Wraps ``fn(request) -> bool`` (e.g. a Slack bot) with a timeout."""

    def __init__(self, fn: Callable[[ApprovalRequest], bool], timeout: float | None = 60.0,
                 name: str = "callback"):
        self.fn, self.timeout, self.name = fn, timeout, name

    def __call__(self, req):
        ok, note = _with_timeout(lambda: self.fn(req), self.timeout)
        return ApprovalOutcome(ok, self.name, note)


class CLIApprover:
    """Asks on the terminal. Only an explicit 'y' / 'yes' approves."""
    name = "cli"

    def __init__(self, timeout: float | None = 120.0, stdin: TextIO | None = None,
                 stdout: TextIO | None = None):
        self.timeout = timeout
        self.stdin, self.stdout = stdin, stdout

    def __call__(self, req):
        out = self.stdout or sys.stderr
        print(f"\n[safejev] approval needed ({req.risk_level}): {req.tool}"
              f"({json.dumps(req.arguments, default=str)})", file=out)
        for r in req.reasons:
            print(f"  - {r}", file=out)
        print("  approve? [y/N] ", end="", file=out, flush=True)
        stdin = self.stdin or sys.stdin
        ok, note = _with_timeout(lambda: stdin.readline().strip().lower() in ("y", "yes"),
                                 self.timeout)
        return ApprovalOutcome(ok, self.name, note)


class WebhookApprover:
    """POSTs the request as JSON; the endpoint answers ``{"approved": true|false}``."""
    name = "webhook"

    def __init__(self, url: str, timeout: float = 30.0, headers: dict[str, str] | None = None):
        self.url, self.timeout, self.headers = url, timeout, dict(headers or {})

    def _post(self, req: ApprovalRequest) -> bool:
        body = json.dumps(asdict(req), default=str).encode()
        r = urllib.request.Request(self.url, data=body, method="POST",
                                   headers={"Content-Type": "application/json", **self.headers})
        with urllib.request.urlopen(r, timeout=self.timeout) as resp:  # noqa: S310 - user URL
            return json.loads(resp.read().decode() or "{}").get("approved") is True

    def __call__(self, req):
        ok, note = _with_timeout(lambda: self._post(req), self.timeout + 1)
        return ApprovalOutcome(ok, self.name, note)
