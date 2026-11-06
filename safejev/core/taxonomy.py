"""Label taxonomy (design doc §7)."""
from __future__ import annotations

DECISIONS: tuple[str, ...] = ("allow", "ask", "block")
SEVERITIES: tuple[str, ...] = ("none", "low", "medium", "high", "critical")
MODULES: tuple[str, ...] = ("guard", "misuse", "agent")
GRANULARITIES: tuple[str, ...] = ("text", "step", "trajectory")

MODULE_TYPES: dict[str, tuple[str, ...]] = {
    "guard": (
        "direct_injection", "indirect_injection", "jailbreak", "context_manipulation",
        "hierarchy_violation", "obfuscated_instruction",
    ),
    "misuse": ("cyber_abuse", "phishing_fraud", "deceptive_content", "general_harm"),
    "agent": (
        "malicious_request", "injected_goal", "harmful_tool", "risky_arguments",
        "intent_mismatch", "unauthorized_action", "data_exfiltration", "destructive_action",
        "malformed_call", "multi_step_risk",
    ),
}

ALL_TYPES: tuple[str, ...] = tuple(t for m in MODULES for t in MODULE_TYPES[m])
TYPE_TO_MODULE: dict[str, str] = {t: m for m in MODULES for t in MODULE_TYPES[m]}

#: Severity a detection of this type gets when the detector does not supply one.
DEFAULT_SEVERITY: dict[str, str] = {
    "direct_injection": "high", "indirect_injection": "high", "jailbreak": "high",
    "context_manipulation": "medium", "hierarchy_violation": "medium",
    "obfuscated_instruction": "medium",
    "cyber_abuse": "high", "phishing_fraud": "high", "deceptive_content": "medium",
    "general_harm": "high",
    "malicious_request": "high", "injected_goal": "critical", "harmful_tool": "high",
    "risky_arguments": "medium", "intent_mismatch": "high", "unauthorized_action": "high",
    "data_exfiltration": "critical", "destructive_action": "high", "malformed_call": "low",
    "multi_step_risk": "high",
}

#: Agent risk induced from outside (malicious user, injection) vs. the agent's own mistakes on
#: benign tasks (HINTBench-style). Reported as separate metric slices (§7).
EXTERNAL_TYPES: frozenset[str] = frozenset({"malicious_request", "injected_goal"})
INTRINSIC_TYPES: frozenset[str] = frozenset(MODULE_TYPES["agent"]) - EXTERNAL_TYPES


def severity_rank(sev: str) -> int:
    try:
        return SEVERITIES.index(sev)
    except ValueError as e:
        raise ValueError(f"unknown severity {sev!r}; expected one of {SEVERITIES}") from e


def decision_rank(dec: str) -> int:
    try:
        return DECISIONS.index(dec)
    except ValueError as e:
        raise ValueError(f"unknown decision {dec!r}; expected one of {DECISIONS}") from e


def max_severity(sevs) -> str:
    sevs = list(sevs)
    return max(sevs, key=severity_rank) if sevs else "none"


def max_decision(decs) -> str:
    decs = list(decs)
    return max(decs, key=decision_rank) if decs else "allow"


def risk_slice(types) -> str:
    """'external', 'intrinsic', 'mixed' or 'none' for a set of agent types."""
    ts = set(types) & set(MODULE_TYPES["agent"])
    ext, intr = bool(ts & EXTERNAL_TYPES), bool(ts & INTRINSIC_TYPES)
    return "mixed" if ext and intr else "external" if ext else "intrinsic" if intr else "none"
