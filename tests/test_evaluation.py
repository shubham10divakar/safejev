import random

import pytest

from safejev import SafeJEV
from safejev.core.detector import ScanInput
from safejev.evaluation import (auprc, auroc, bench, bootstrap_ci, decision_report, early_catch,
                                leave_one_out_plan, predict, recall_at_fpr, report, to_markdown)
from safejev.evaluation.harness import Prediction
from training import mutations


def test_auroc_auprc_known_values():
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert auroc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert auroc([1, 2], [1, 1]) is None
    assert auprc([0.9, 0.8, 0.7], [1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)
    assert auprc([1.0], [0]) is None


def test_recall_at_fpr():
    neg = [i / 100 for i in range(100)]
    pos = [0.995, 0.985, 0.5]
    s, y = neg + pos, [0] * 100 + [1] * 3
    assert recall_at_fpr(s, y, 0.0) == pytest.approx(1 / 3)      # above max neg 0.99
    assert recall_at_fpr(s, y, 0.01) == pytest.approx(2 / 3)     # one neg allowed above
    assert recall_at_fpr(s, y, 0.6) == 1.0


def test_decision_report_and_bootstrap():
    d = decision_report(["allow", "ask", "block", "allow"], ["allow", "ask", "ask", "block"])
    assert d["accuracy"] == 0.5 and d["benign_completion"] == 1.0 and d["miss_rate"] == pytest.approx(1 / 3)
    rng = random.Random(0)
    y = [rng.randint(0, 1) for _ in range(200)]
    s = [v + rng.gauss(0, 0.7) for v in y]
    pt, lo, hi = bootstrap_ci(auroc, s, y, n=200)
    assert lo < pt < hi and 0.6 < pt < 0.95


def test_harness_on_mutations_with_heuristics():
    recs = list(mutations.generate(mutations.builtin_seeds(random.Random(0), 3), random.Random(0)))
    sj = SafeJEV(modules=["agent"], backend="heuristic")
    preds = predict(sj, recs)
    rep = report(preds, n_boot=50)
    assert rep["n"] == len(recs) and rep["risk"]["auroc"] > 0.6
    assert "intent_mismatch" in rep["per_type"] and "benign" in rep["slices"]
    assert "destructive_action_rate" in rep["agent_rates"]
    md = to_markdown(rep)
    assert "risk auroc" in md and "| intent_mismatch |" in md


def test_early_catch_and_loo():
    def p(i, dec):
        return Prediction(f"t1-step{i}", dec, 0, {}, "allow", [], "R-Judge", "none", 0,
                          {"step_index": i, "risky_step_index": 2})
    ec = early_catch([p(0, "allow"), p(1, "ask"), p(2, "block")])
    assert ec == {"trajectories": 1, "missed": 0, "mean_offset": -1.0, "early": 1, "on_time": 0, "late": 0}
    recs = list(mutations.generate(mutations.builtin_seeds(random.Random(0), 1), random.Random(0)))
    plan = leave_one_out_plan(recs)
    assert len(plan) == 5 and plan[-1]["n_train"] == len(recs)


def test_latency_bench():
    sj = SafeJEV(modules=["guard", "agent"], backend="heuristic")
    inps = [ScanInput(user_input="find files", agent_action={"tool": "list_files", "arguments": {"path": "/a"}})]
    b = bench(sj, inps, repeat=5)
    assert b["n"] == 5 and b["p95_ms"] >= b["p50_ms"] > 0 and b["adapters"] == 3
    with pytest.raises(ValueError):
        bench(sj, [])
