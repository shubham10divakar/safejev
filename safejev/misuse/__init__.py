"""Misuse by malicious actors: cyber abuse, phishing / fraud, deceptive content (§2 #3).

Detects *intent to generate* harmful or deceptive content; it does not fact-check.
"""
from ..detectors.heuristic import MisuseDetector

ADAPTERS = ("misuse-request",)
__all__ = ["ADAPTERS", "MisuseDetector"]
