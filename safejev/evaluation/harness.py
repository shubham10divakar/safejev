"""Evaluation harness (design doc §11): run a SafeJEV engine over unified Records and report
the common metrics, per-type scores, risk slices and bootstrap CIs.

Used for E1 (configs compared on the same test sets), E2 (``leave_one_out_plan``), E4 (adapters
trained with different ``last_k``; ``early_catch`` for how early risk is caught), E6 (ECE in
every report) and the baselines (any object with the Detector interface can be plugged in).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from ..core.detector import ScanInput
from ..core.schema import Record
from ..core.taxonomy import MODULE_TYPES, risk_slice
from .bootstrap import bootstrap_ci
from .metrics import auroc, binary_report, decision_report, rate_allowed


@dataclass
class Prediction:
    id: str
    decision: str
    score: float                       # max type score (risk score)
    type_scores: dict[str, float]
    gold_decision: str
    gold_types: list[str]
    dataset: str
    slice: str
    latency_ms: float
    meta: dict = field(default_factory=dict)


def _scan_input(r: Record) -> ScanInput:
    if r.granularity == "text":
        return ScanInput(user_input=r.input.text, untrusted_context=list(r.input.untrusted_context))
    return ScanInput.from_record(r)


def predict(sj, records: Iterable[Record]) -> list[Prediction]:
    out = []
    for r in records:
        inp = _scan_input(r)
        res = sj.scan_input(inp, audit=False)
        ts = dict(res.scores)              # all type scores, fired or not
        out.append(Prediction(r.id, res.decision, max(ts.values(), default=0.0), ts,
                              r.label.decision, list(r.label.types), r.source.dataset,
                              risk_slice(r.label.types), res.latency_ms, dict(r.meta)))
    return out


def report(preds: list[Prediction], n_boot: int = 1000, seed: int = 0) -> dict:
    labels = [int(p.gold_decision != "allow") for p in preds]
    scores = [p.score for p in preds]
    pred_dec = [p.decision for p in preds]
    gold_dec = [p.gold_decision for p in preds]
    rep: dict = {"n": len(preds), "risk": binary_report(scores, labels),
                 "decision": decision_report(pred_dec, gold_dec)}
    if n_boot:
        rep["risk"]["auroc_ci"] = bootstrap_ci(auroc, scores, labels, n=n_boot, seed=seed)[1:]
    # flagged = not allowed, as a binary detector
    rep["flagged"] = binary_report([float(d != "allow") for d in pred_dec], labels)
    per_type = {}
    all_types = sorted({t for p in preds for t in p.type_scores} | {t for p in preds for t in p.gold_types})
    for t in all_types:
        y = [int(t in p.gold_types) for p in preds]
        if any(y):
            s = [p.type_scores.get(t, 0.0) for p in preds]
            per_type[t] = {"auroc": auroc(s, y), "n_pos": sum(y)}
    rep["per_type"] = per_type
    agent_types = MODULE_TYPES["agent"]
    golds = [p.gold_types for p in preds]
    rep["agent_rates"] = {f"{t}_rate": rate_allowed(pred_dec, golds, t)
                          for t in ("unauthorized_action", "data_exfiltration", "destructive_action")
                          if any(t in g for g in golds)}
    slices = defaultdict(list)
    for p in preds:
        if any(t in agent_types for t in p.gold_types):
            slices[p.slice].append(p)
    benign = [p for p in preds if p.gold_decision == "allow"]
    rep["slices"] = {k: {"n": len(v), "recall": sum(x.decision != "allow" for x in v) / len(v)}
                     for k, v in slices.items()}
    if benign:
        rep["slices"]["benign"] = {"n": len(benign),
                                   "fpr": sum(x.decision != "allow" for x in benign) / len(benign)}
    by_ds = defaultdict(list)
    for p in preds:
        by_ds[p.dataset].append(p)
    if len(by_ds) > 1:
        rep["by_dataset"] = {k: report(v, n_boot=0) for k, v in by_ds.items()}
    lat = sorted(p.latency_ms for p in preds)
    if lat:
        rep["latency_ms"] = {"p50": lat[len(lat) // 2], "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))]}
    return rep


def early_catch(preds: list[Prediction]) -> dict:
    """E4: per trajectory, the first flagged step index vs the annotated risky step.

    Needs step predictions whose meta has ``trajectory_id`` (or derived_from), ``step_index`` and
    ``risky_step_index``."""
    by_traj = defaultdict(list)
    for p in preds:
        tid = p.meta.get("trajectory_id") or p.id.rsplit("-step", 1)[0]
        if p.meta.get("risky_step_index") is not None:
            by_traj[tid].append(p)
    deltas, missed = [], 0
    for steps in by_traj.values():
        steps.sort(key=lambda p: p.meta["step_index"])
        risky = steps[0].meta["risky_step_index"]
        first = next((p.meta["step_index"] for p in steps if p.decision != "allow"), None)
        if first is None:
            missed += 1
        else:
            deltas.append(first - risky)
    return {"trajectories": len(by_traj), "missed": missed,
            "mean_offset": sum(deltas) / len(deltas) if deltas else None,
            "early": sum(d < 0 for d in deltas), "on_time": sum(d == 0 for d in deltas),
            "late": sum(d > 0 for d in deltas)}


#: E2 leave-one-benchmark-out plan (design doc §11).
LEAVE_ONE_OUT = [
    {"held_out": "Pre-Exec-Bench", "train_excludes": ["Pre-Exec-Bench"]},
    {"held_out": "R-Judge", "train_excludes": ["R-Judge"]},
    {"held_out": "HINTBench", "train_excludes": ["HINTBench"]},
    {"held_out": "AgentDojo-Traj", "train_excludes": ["AgentDojo", "AgentDojo-Traj"]},
    {"held_out": "all-real", "train_only": ["synthetic-mutations", "AuraGen"]},
]


def leave_one_out_plan(train_pool: list[Record]) -> list[dict]:
    """For each E2 row: which train records are allowed (ids) and the held-out benchmark."""
    out = []
    for row in LEAVE_ONE_OUT:
        if "train_only" in row:
            keep = [r for r in train_pool if r.source.dataset in row["train_only"]]
        else:
            keep = [r for r in train_pool if r.source.dataset not in row["train_excludes"]]
        out.append({"held_out": row["held_out"], "n_train": len(keep), "train_ids": [r.id for r in keep]})
    return out


def to_markdown(rep: dict, title: str = "SafeJEV evaluation") -> str:
    def f(x):
        return "—" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
    r, d = rep["risk"], rep["decision"]
    lines = [f"# {title}", "", f"n = {rep['n']}", "", "| metric | value |", "|---|---|"]
    for k in ("auroc", "auprc", "recall@fpr1", "recall@fpr5", "f1", "ece"):
        lines.append(f"| risk {k} | {f(r.get(k))} |")
    if r.get("auroc_ci"):
        lines.append(f"| risk AUROC 95% CI | {f(r['auroc_ci'][0])} – {f(r['auroc_ci'][1])} |")
    for k in ("accuracy", "macro_f1", "approval_frequency", "benign_completion", "miss_rate"):
        lines.append(f"| decision {k} | {f(d.get(k))} |")
    if rep.get("per_type"):
        lines += ["", "| type | AUROC | n_pos |", "|---|---|---|"]
        lines += [f"| {t} | {f(v['auroc'])} | {v['n_pos']} |" for t, v in rep["per_type"].items()]
    if rep.get("slices"):
        lines += ["", "```json", json.dumps(rep["slices"], indent=1), "```"]
    return "\n".join(lines) + "\n"
