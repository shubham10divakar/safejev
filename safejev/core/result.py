"""Detection and scan-result types returned by the engine (design doc §8)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Detection:
    module: str
    type: str
    score: float                 # probability that this type is present
    adapter: str                 # "safejev/name@version"
    severity: str = "medium"
    threshold: float = 0.5       # the adapter's own threshold; fired iff score >= threshold
    reason: str = ""

    @property
    def fired(self) -> bool:
        return self.score >= self.threshold


@dataclass
class ScanResult:
    decision: str                       # allow / ask / block (after policy)
    risk_level: str                     # none ... critical (from fusion)
    detections: list[Detection] = field(default_factory=list)   # fired detections only
    latency_ms: float = 0.0
    policy_rule: str = ""               # which policy rule produced the decision
    adapters: list[str] = field(default_factory=list)           # name@version that ran
    errors: list[str] = field(default_factory=list)             # detector failures
    repair_hint: str = ""
    reasons: list[str] = field(default_factory=list)             # human-readable why
    scores: dict[str, float] = field(default_factory=dict)       # max score per type, fired or not

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for det in d["detections"]:
            det["score"] = round(det["score"], 4)
        d["latency_ms"] = round(self.latency_ms, 2)
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)
