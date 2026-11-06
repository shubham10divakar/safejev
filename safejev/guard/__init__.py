"""Inference-time prompt attacks: injection (direct / indirect), jailbreak (design doc §2 #1)."""
from ..detectors.heuristic import InjectionDetector, JailbreakDetector

ADAPTERS = ("guard-injection", "guard-jailbreak")
__all__ = ["ADAPTERS", "InjectionDetector", "JailbreakDetector"]
