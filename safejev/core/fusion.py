"""Fusion (design doc §5.4): each adapter's score against its own threshold; the
highest-severity fired detection sets the risk level. No learned router in v0.1."""
from __future__ import annotations

from dataclasses import dataclass

from .result import Detection
from .taxonomy import max_severity, severity_rank


@dataclass
class Fused:
    risk_level: str
    fired: list[Detection]        # sorted: highest severity first, then score
    top: Detection | None


def fuse(detections: list[Detection]) -> Fused:
    fired = [d for d in detections if d.fired]
    fired.sort(key=lambda d: (severity_rank(d.severity), d.score), reverse=True)
    return Fused(risk_level=max_severity(d.severity for d in fired), fired=fired,
                 top=fired[0] if fired else None)
