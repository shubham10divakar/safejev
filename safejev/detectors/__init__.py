"""Detector implementations: heuristic baselines (no weights) and the encoder adapter."""
from __future__ import annotations

from .heuristic import InjectionDetector, JailbreakDetector, MisuseDetector
from .precheck import PrecheckDetector

HEURISTICS = {
    "guard-injection": InjectionDetector,
    "guard-jailbreak": JailbreakDetector,
    "misuse-request": MisuseDetector,
    "agent-precheck": PrecheckDetector,
}


def heuristic(name: str, **kw):
    """Build the heuristic detector for adapter ``name``."""
    if name not in HEURISTICS:
        raise KeyError(f"no heuristic implementation for {name!r}; available: {sorted(HEURISTICS)}")
    return HEURISTICS[name](**kw)


__all__ = ["HEURISTICS", "heuristic", "InjectionDetector", "JailbreakDetector",
           "MisuseDetector", "PrecheckDetector"]
