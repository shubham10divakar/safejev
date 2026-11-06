"""Deterministic policy (design doc §12): fused detections + authorization → allow / ask / block.

Order of evaluation; the most restrictive outcome wins, and the rule that produced it is
reported:

1. authorization violations (allowlist, argument constraints, rate limits) → ``authz_decision``
2. fail-closed: a detector failed and the tool is sensitive → ``fail_closed_decision``
3. explicit rules: (module, type, severity, tool) → decision
4. severity thresholds: ``block_on`` / ``ask_on`` against the fused risk level
5. ``require_approval_tools``: those tools are always at least ``ask``
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..core.detector import ScanInput
from ..core.fusion import Fused
from ..core.taxonomy import DECISIONS, SEVERITIES, decision_rank, severity_rank
from .authz import Authorizer


@dataclass
class PolicyRule:
    """Fires when every set field matches a fired detection (tool may also match alone)."""
    decision: str
    name: str = ""
    module: str | None = None
    type: str | None = None
    min_severity: str | None = None
    tool: str | None = None            # fnmatch pattern
    min_score: float | None = None

    def __post_init__(self):
        if self.decision not in DECISIONS:
            raise ValueError(f"rule decision {self.decision!r} not in {DECISIONS}")
        if self.min_severity is not None and self.min_severity not in SEVERITIES:
            raise ValueError(f"rule severity {self.min_severity!r} not in {SEVERITIES}")
        if not self.name:
            parts = [f"{k}={v}" for k, v in (("module", self.module), ("type", self.type),
                                             ("severity>=", self.min_severity), ("tool", self.tool))
                     if v]
            self.name = "rule:" + ",".join(parts) + f"->{self.decision}"

    def matches(self, fused: Fused, tool: str) -> bool:
        if self.tool is not None and not fnmatch.fnmatchcase(tool or "", self.tool):
            return False
        needs_detection = any(v is not None for v in (self.module, self.type, self.min_severity,
                                                      self.min_score))
        if not needs_detection:
            return self.tool is not None
        for d in fused.fired:
            if self.module is not None and d.module != self.module:
                continue
            if self.type is not None and d.type != self.type:
                continue
            if self.min_severity is not None and severity_rank(d.severity) < severity_rank(self.min_severity):
                continue
            if self.min_score is not None and d.score < self.min_score:
                continue
            return True
        return False


@dataclass
class PolicyDecision:
    decision: str
    rule: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class Policy:
    block_on: list[str] = field(default_factory=lambda: ["critical"])
    ask_on: list[str] = field(default_factory=lambda: ["high"])
    require_approval_tools: list[str] = field(default_factory=list)    # fnmatch patterns
    sensitive_tools: list[str] = field(default_factory=list)           # for fail-closed
    fail_closed: bool = True
    fail_closed_decision: str = "block"
    authz: Authorizer | None = None
    authz_decision: str = "block"
    rules: list[PolicyRule] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)         # per adapter name
    audit_log: str | None = None
    audit_store_inputs: bool = False
    approval_timeout: float = 60.0

    def __post_init__(self):
        for s in list(self.block_on) + list(self.ask_on):
            severity_rank(s)
        for d in (self.fail_closed_decision, self.authz_decision):
            decision_rank(d)
        self.rules = [r if isinstance(r, PolicyRule) else PolicyRule(**r) for r in self.rules]
        if isinstance(self.authz, dict):
            self.authz = Authorizer.from_dict(self.authz)

    # ---- loading ---------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Policy":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown policy keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Policy":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    # ---- evaluation ------------------------------------------------------------------------
    def is_sensitive(self, tool: str, declared: bool = False) -> bool:
        pats = list(self.sensitive_tools) + list(self.require_approval_tools)
        return declared or any(fnmatch.fnmatchcase(tool, p) for p in pats)

    def decide(self, fused: Fused, inp: ScanInput, errors: list[str] | None = None,
               sensitive: bool = False, record_rate: bool = True) -> PolicyDecision:
        tool = inp.tool_name
        cands: list[PolicyDecision] = [PolicyDecision("allow", "default")]

        if self.authz is not None and tool:
            viol = self.authz.check(tool, inp.arguments, record=record_rate)
            if viol:
                cands.append(PolicyDecision(self.authz_decision, "authz", viol))

        if errors and self.fail_closed and tool and self.is_sensitive(tool, sensitive):
            cands.append(PolicyDecision(self.fail_closed_decision, "fail_closed", list(errors)))

        for r in self.rules:
            if r.matches(fused, tool):
                cands.append(PolicyDecision(r.decision, r.name))

        if fused.risk_level in self.block_on:
            cands.append(PolicyDecision("block", f"block_on:{fused.risk_level}"))
        elif fused.risk_level in self.ask_on:
            cands.append(PolicyDecision("ask", f"ask_on:{fused.risk_level}"))

        if tool and any(fnmatch.fnmatchcase(tool, p) for p in self.require_approval_tools):
            cands.append(PolicyDecision("ask", "require_approval_tools"))

        best = max(cands, key=lambda c: decision_rank(c.decision))   # first max wins ties
        reasons = list(best.reasons) + [f"{d.type} ({d.score:.2f}): {d.reason}" if d.reason
                                        else f"{d.type} ({d.score:.2f})" for d in fused.fired]
        return PolicyDecision(best.decision, best.rule, reasons)
