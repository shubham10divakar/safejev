"""Deterministic authorization (design doc §12): per-tool allowlists, argument constraints and
rate limits. These run on every call, even when every detector says ``allow``."""
from __future__ import annotations

import fnmatch
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse


@dataclass
class ArgConstraint:
    """Constraint on one argument. Unset fields are not checked."""
    allowed_prefixes: list[str] | None = None     # paths
    denied_prefixes: list[str] | None = None
    allowed_domains: list[str] | None = None      # URLs or e-mail addresses
    one_of: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    regex: str | None = None
    required: bool = False

    def check(self, name: str, present: bool, value: Any) -> list[str]:
        if not present:
            return [f"argument {name!r} is required"] if self.required else []
        errs = []
        sv = str(value)
        if self.allowed_prefixes is not None and not any(sv.startswith(p) for p in self.allowed_prefixes):
            errs.append(f"{name}={sv!r} is outside the allowed prefixes {self.allowed_prefixes}")
        if self.denied_prefixes and any(sv.startswith(p) for p in self.denied_prefixes):
            errs.append(f"{name}={sv!r} is under a denied prefix")
        if self.allowed_domains is not None:
            dom = _domain(sv)
            if not dom or not any(dom == d or dom.endswith("." + d) for d in self.allowed_domains):
                errs.append(f"{name}={sv!r}: domain {dom!r} is not allowed")
        if self.one_of is not None and value not in self.one_of:
            errs.append(f"{name}={value!r} is not one of {self.one_of}")
        if self.min is not None or self.max is not None:
            try:
                x = float(value)
            except (TypeError, ValueError):
                errs.append(f"{name}={value!r} is not a number")
            else:
                if self.min is not None and x < self.min:
                    errs.append(f"{name}={x} is below {self.min}")
                if self.max is not None and x > self.max:
                    errs.append(f"{name}={x} is above {self.max}")
        if self.regex is not None and not re.fullmatch(self.regex, sv):
            errs.append(f"{name}={sv!r} does not match {self.regex!r}")
        return errs


def _domain(s: str) -> str:
    if "@" in s and "://" not in s:
        return s.rsplit("@", 1)[-1].lower()
    host = urlparse(s if "://" in s else "//" + s).hostname
    return (host or "").lower()


@dataclass
class RateLimit:
    calls: int
    per_seconds: float

    @classmethod
    def parse(cls, spec: str | dict | "RateLimit") -> "RateLimit":
        """'10/60s', '5/m', '100/h' or {'calls': 10, 'per_seconds': 60}."""
        if isinstance(spec, RateLimit):
            return spec
        if isinstance(spec, dict):
            return cls(int(spec["calls"]), float(spec["per_seconds"]))
        m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d*\.?\d*)\s*(s|m|h)?\s*", spec)
        if not m:
            raise ValueError(f"bad rate limit {spec!r}; use e.g. '10/60s' or '5/m'")
        n, num, unit = int(m.group(1)), float(m.group(2) or 1), m.group(3) or "s"
        return cls(n, num * {"s": 1, "m": 60, "h": 3600}[unit])


@dataclass
class ToolRule:
    tool: str                                     # fnmatch pattern, e.g. "delete_*"
    allow: bool = True
    args: dict[str, ArgConstraint] = field(default_factory=dict)
    rate_limit: RateLimit | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ToolRule":
        return cls(tool=d["tool"], allow=d.get("allow", True),
                   args={k: ArgConstraint(**v) for k, v in (d.get("args") or {}).items()},
                   rate_limit=RateLimit.parse(d["rate_limit"]) if d.get("rate_limit") else None)


class Authorizer:
    """``check(tool, args)`` returns a list of violations (empty = authorized).

    ``default_allow=False`` turns the rule list into an allowlist: a tool no rule matches is
    denied.
    """

    def __init__(self, rules: list[ToolRule] | None = None, default_allow: bool = True,
                 clock: Callable[[], float] = time.monotonic):
        self.rules = list(rules or [])
        self.default_allow = default_allow
        self._clock = clock
        self._calls: dict[str, deque] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_dict(cls, d: dict) -> "Authorizer":
        return cls([ToolRule.from_dict(r) for r in d.get("tools", [])],
                   default_allow=d.get("default_allow", True))

    def check(self, tool: str, args: dict[str, Any], record: bool = True) -> list[str]:
        matched = [r for r in self.rules if fnmatch.fnmatchcase(tool, r.tool)]
        if not matched:
            return [] if self.default_allow else [f"tool {tool!r} is not on the allowlist"]
        errs: list[str] = []
        for r in matched:
            if not r.allow:
                errs.append(f"tool {tool!r} is denied by rule {r.tool!r}")
            for name, c in r.args.items():
                errs.extend(c.check(name, name in args, args.get(name)))
            if r.rate_limit:
                errs.extend(self._rate(tool, r, record and not errs))
        return errs

    def _rate(self, tool: str, r: ToolRule, record: bool) -> list[str]:
        now = self._clock()
        key = f"{r.tool}::{tool}"
        with self._lock:
            q = self._calls.setdefault(key, deque())
            while q and now - q[0] >= r.rate_limit.per_seconds:
                q.popleft()
            if len(q) >= r.rate_limit.calls:
                return [f"rate limit {r.rate_limit.calls}/{r.rate_limit.per_seconds:g}s exceeded for {tool!r}"]
            if record:
                q.append(now)
        return []
