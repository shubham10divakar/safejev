"""Convert benchmark rows to unified Records (design doc §6, §7, §9.3).

The field names of the public benchmarks below are **assumptions** until checked against the
downloaded files (none are downloaded in this round). Each converter takes plain dicts, so a
mismatch is fixed in one place. Original labels are always kept in ``original_label``.
"""
from __future__ import annotations

from typing import Any, Iterator

from safejev.adapters_io.trajectory import normalize_action
from safejev.core.schema import Label, Record, RecordInput, Source
from safejev.core.taxonomy import DEFAULT_SEVERITY, max_severity

TSBENCH_DECISION = {"safe": "allow", "controversial": "ask", "unsafe": "block"}


def _label(decision: str, types: list[str]) -> Label:
    if decision == "allow":
        return Label("allow", [], "none")
    sev = max_severity(DEFAULT_SEVERITY[t] for t in types) if types else "high"
    return Label(decision, list(types), sev)


def tsbench_step(row: dict[str, Any], split: str, idx: int, type_map: dict[str, str] | None = None
                 ) -> Record:
    """Assumed fields: instruction / user_goal, history (list of steps), action
    ({"name"/"tool", "arguments"}), label in {safe, controversial, unsafe}, optional
    risk_type."""
    raw = str(row["label"]).lower()
    decision = TSBENCH_DECISION[raw]
    rt = row.get("risk_type")
    types = [] if decision == "allow" else [(type_map or {}).get(rt, rt or "multi_step_risk")]
    return Record(
        id=f"tsbench-{split}-{idx:06d}",
        source=Source("TS-Bench", split, row.get("version", "v1"), row.get("license", "unknown")),
        granularity="step", module="agent",
        input=RecordInput(user_goal=row.get("user_goal") or row.get("instruction", ""),
                          history=list(row.get("history") or []),
                          proposed_action=normalize_action(row["action"]),
                          tool_schema=row.get("tool_schema")),
        label=_label(decision, types),
        original_label={"raw": raw, "scheme": "TS-Bench safe/controversial/unsafe",
                        "risk_type": rt},
        meta={"scenario": row.get("scenario"), "domain": row.get("domain")},
    )


def trajectory_record(dataset: str, split: str, idx: int, goal: str, steps: list[dict],
                      unsafe: bool, types: list[str] | None = None, raw_label: Any = None,
                      license: str = "unknown", meta: dict | None = None) -> Record:
    """A whole trajectory as one record (R-Judge's binary label, HINTBench)."""
    types = list(types or (["multi_step_risk"] if unsafe else []))
    last = next((normalize_action(s["content"]) for s in reversed(steps)
                 if s.get("role") == "tool_call"), {"tool": "", "arguments": {}})
    return Record(
        id=f"{dataset.lower()}-{split}-{idx:06d}", source=Source(dataset, split, "", license),
        granularity="trajectory", module="agent",
        input=RecordInput(user_goal=goal, history=list(steps), proposed_action=last),
        label=_label("block" if unsafe else "allow", types if unsafe else []),
        original_label={"raw": raw_label, "scheme": f"{dataset} binary"}, meta=dict(meta or {}))


def trajectory_to_steps(rec: Record, risky_step: int | None) -> Iterator[Record]:
    """§9.3: cut at every tool call. Calls before ``risky_step`` (index into history, from an
    LLM pass verified by hand) are ``allow``; the risky call gets the trajectory's label; calls
    after it are dropped (their context is already compromised). If the trajectory is safe,
    every call is ``allow``. ``risky_step=None`` on an unsafe trajectory yields nothing,
    because the step label is unknown."""
    unsafe = rec.label.decision != "allow"
    if unsafe and risky_step is None:
        return
    history: list[dict] = []
    n = 0
    for i, s in enumerate(rec.input.history):
        if s.get("role") == "tool_call":
            action = normalize_action(s["content"])
            if unsafe and i > risky_step:
                return
            is_risky = unsafe and i == risky_step
            lab = rec.label if is_risky else Label("allow", [], "none")
            yield Record(
                id=f"{rec.id}-step{i}", source=rec.source, granularity="step", module="agent",
                input=RecordInput(user_goal=rec.input.user_goal, history=list(history),
                                  proposed_action=action),
                label=Label(lab.decision, list(lab.types), lab.severity),
                original_label=dict(rec.original_label, derived_from=rec.id),
                meta=dict(rec.meta, step_index=i, risky_step_index=risky_step,
                          human_verified=bool(rec.meta.get("risky_step_verified"))))
            n += 1
        history.append(s)


def rjudge(row: dict[str, Any], split: str, idx: int) -> Record:
    """Assumed fields: profile/goal, contents (list of {role, content}), label (0/1),
    risk_description."""
    steps = []
    for turn in row.get("contents", []):
        for s in (turn if isinstance(turn, list) else [turn]):
            role = s.get("role", "")
            if role == "agent" and s.get("action"):
                steps.append({"role": "tool_call", "content": s["action"]})
            elif role in ("environment", "tool"):
                steps.append({"role": "tool_result", "content": s.get("content", "")})
            else:
                steps.append({"role": role or "user", "content": s.get("content", "")})
    goal = row.get("goal") or next((s["content"] for s in steps if s["role"] == "user"), "")
    return trajectory_record("R-Judge", split, idx, goal, steps, bool(int(row["label"])),
                             raw_label=row["label"],
                             meta={"scenario": row.get("scenario"),
                                   "risk_description": row.get("risk_description")})


def text_record(dataset: str, split: str, idx: int, module: str, text: str, decision: str,
                types: list[str], raw_label: Any = None, license: str = "unknown",
                meta: dict | None = None) -> Record:
    """Prompt-level sources (guard / misuse)."""
    return Record(id=f"{dataset.lower()}-{split}-{idx:06d}", source=Source(dataset, split, "", license),
                  granularity="text", module=module, input=RecordInput(text=text),
                  label=_label(decision, types if decision != "allow" else []),
                  original_label={"raw": raw_label}, meta=dict(meta or {}))
