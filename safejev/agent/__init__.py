"""Intrinsic and induced risks of autonomous agents: the pre-execution tool-call check (§2 #4)."""
from ..detectors.precheck import PrecheckDetector

ADAPTERS = ("agent-precheck", "agent-trajectory")
__all__ = ["ADAPTERS", "PrecheckDetector"]
