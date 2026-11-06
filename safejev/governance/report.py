"""Summaries over an audit log (design doc §12): detections by type, approval load, and an
FPR estimate from blocks/asks that a human overrode."""
from __future__ import annotations

from collections import Counter
from typing import Iterable


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(entries: Iterable[dict]) -> dict:
    entries = list(entries)
    n = len(entries)
    decisions = Counter(e.get("decision") for e in entries)
    types = Counter(d["type"] for e in entries for d in e.get("detections", []))
    tools = Counter(e.get("tool") for e in entries if e.get("tool"))
    rules = Counter(e.get("policy_rule") for e in entries if e.get("policy_rule"))
    asked = [e for e in entries if "approval" in e]
    approved = sum(e["approval"]["approved"] for e in asked)
    lat = [e.get("latency_ms", 0.0) for e in entries]
    return {
        "total": n,
        "decisions": dict(decisions),
        "by_type": dict(types.most_common()),
        "by_tool": dict(tools.most_common()),
        "by_rule": dict(rules.most_common()),
        "approval_requests": len(asked),
        "approval_frequency": len(asked) / n if n else 0.0,
        "approved": approved,
        "denied": len(asked) - approved,
        # an approved request means a human judged the flag unnecessary: upper bound on FPR
        "fpr_estimate_from_overrides": approved / len(asked) if asked else None,
        "errors": sum(bool(e.get("errors")) for e in entries),
        "latency_ms": {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95)},
    }


def format_markdown(s: dict) -> str:
    lines = ["# SafeJEV audit report", "", f"- entries: **{s['total']}**"]
    for k in ("allow", "ask", "block"):
        lines.append(f"- {k}: {s['decisions'].get(k, 0)}")
    lines.append(f"- approval frequency: {s['approval_frequency']:.1%} "
                 f"({s['approved']} approved / {s['denied']} denied)")
    if s["fpr_estimate_from_overrides"] is not None:
        lines.append(f"- FPR estimate from overrides: {s['fpr_estimate_from_overrides']:.1%}")
    lines.append(f"- detector errors: {s['errors']}")
    lines.append(f"- latency p50 / p95: {s['latency_ms']['p50']:.1f} / {s['latency_ms']['p95']:.1f} ms")
    for title, key in (("Detections by type", "by_type"), ("Tools", "by_tool"),
                       ("Policy rules hit", "by_rule")):
        if s[key]:
            lines += ["", f"## {title}", "", "| name | count |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in s[key].items()]
    return "\n".join(lines) + "\n"
