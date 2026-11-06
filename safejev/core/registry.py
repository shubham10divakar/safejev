"""Adapter registry (design doc §5.2): user-facing capability adapters and research adapters.

``heuristic=True`` means a deterministic rule implementation ships in this package, used when
no trained weights are available (and as the baseline in every experiment).
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_BACKBONE = "safejev/backbone-base"


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    module: str
    tier: str            # "user" or "research"
    description: str
    granularity: str = "step"
    heuristic: bool = True


_RESEARCH_SOURCES = ("tsbench", "hintbench", "auragen", "rjudge", "synthetic-mutations")

ADAPTERS: dict[str, AdapterInfo] = {a.name: a for a in [
    AdapterInfo("guard-injection", "guard", "user",
                "Direct + indirect prompt injection, instruction-hierarchy violation", "text"),
    AdapterInfo("guard-jailbreak", "guard", "user",
                "Attempts to bypass safety constraints", "text"),
    AdapterInfo("misuse-request", "misuse", "user",
                "Harmful requests incl. cyber abuse, phishing/fraud, deceptive content", "text"),
    AdapterInfo("agent-precheck", "agent", "user",
                "Pre-execution tool-call check: risky args, harmful tool, injected-goal call, "
                "intent mismatch, malformed call", "step"),
    AdapterInfo("agent-trajectory", "agent", "user",
                "Multi-step risk across a trajectory (decoder variant, planned for v0.3)",
                "trajectory", heuristic=False),
    *[AdapterInfo(f"agent-{src}", "agent", "research",
                  f"agent-precheck trained on {src} only (ablations E1/E2)", "step",
                  heuristic=False)
      for src in _RESEARCH_SOURCES],
]}

DEFAULT_ADAPTERS: dict[str, tuple[str, ...]] = {
    "guard": ("guard-injection", "guard-jailbreak"),
    "misuse": ("misuse-request",),
    "agent": ("agent-precheck",),
}


def parse_ref(ref: str) -> tuple[str, str | None]:
    """'safejev/agent-precheck@0.1.0' -> ('agent-precheck', '0.1.0')."""
    name, _, ver = ref.partition("@")
    return name.removeprefix("safejev/"), (ver or None)


def get(ref: str) -> AdapterInfo:
    name, _ = parse_ref(ref)
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter {ref!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[name]


def list_adapters(tier: str | None = None) -> list[AdapterInfo]:
    return [a for a in ADAPTERS.values() if tier is None or a.tier == tier]


def defaults_for(modules) -> list[str]:
    out: list[str] = []
    for m in modules:
        if m not in DEFAULT_ADAPTERS:
            raise KeyError(f"unknown module {m!r}; known: {sorted(DEFAULT_ADAPTERS)}")
        out.extend(DEFAULT_ADAPTERS[m])
    return out
