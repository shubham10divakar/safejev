"""Common metrics (design doc §11). Pure Python; inputs are plain lists.

Binary metrics take ``scores`` (higher = riskier) and ``labels`` (1 = risky).
"""
from __future__ import annotations

from typing import Sequence

from ..core.calibration import ece


def confusion(pred: Sequence[int], labels: Sequence[int]) -> tuple[int, int, int, int]:
    tp = sum(p and y for p, y in zip(pred, labels))
    fp = sum(p and not y for p, y in zip(pred, labels))
    fn = sum((not p) and y for p, y in zip(pred, labels))
    tn = sum((not p) and (not y) for p, y in zip(pred, labels))
    return tp, fp, fn, tn


def prf(pred, labels) -> dict[str, float]:
    tp, fp, fn, tn = confusion(pred, labels)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": p, "recall": r, "f1": f, "fpr": fpr}


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Mann-Whitney U with average ranks for ties. None if only one class is present."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    r_pos = sum(r for r, y in zip(ranks, labels) if y)
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auprc(scores, labels) -> float | None:
    """Average precision (step-wise area under the PR curve), ties handled as one threshold."""
    n_pos = sum(labels)
    if not n_pos:
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    ap, prev_r = 0.0, 0.0
    i = 0
    while i < len(pairs):
        s = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == s:
            tp += pairs[i][1]
            fp += 1 - pairs[i][1]
            i += 1
        r = tp / n_pos
        ap += (r - prev_r) * (tp / (tp + fp))
        prev_r = r
    return ap


def recall_at_fpr(scores, labels, max_fpr: float) -> float | None:
    """Best recall over thresholds whose FPR on negatives is <= max_fpr."""
    neg = sorted((s for s, y in zip(scores, labels) if not y), reverse=True)
    pos = [s for s, y in zip(scores, labels) if y]
    if not neg or not pos:
        return None
    k = int(max_fpr * len(neg))                 # negatives allowed above the threshold
    # threshold strictly above the (k+1)-th highest negative score
    thr = neg[k] if k < len(neg) else float("-inf")
    return sum(s > thr for s in pos) / len(pos)


def binary_report(scores, labels, threshold: float = 0.5) -> dict:
    pred = [int(s >= threshold) for s in scores]
    out = prf(pred, labels)
    out.update({"auroc": auroc(scores, labels), "auprc": auprc(scores, labels),
                "recall@fpr1": recall_at_fpr(scores, labels, 0.01),
                "recall@fpr5": recall_at_fpr(scores, labels, 0.05),
                "ece": ece(scores, labels), "n": len(labels), "n_pos": int(sum(labels))})
    return out


def decision_report(pred: Sequence[str], gold: Sequence[str]) -> dict:
    """3-way allow/ask/block: accuracy, macro-F1, and the operational rates."""
    from ..core.taxonomy import DECISIONS
    acc = sum(p == g for p, g in zip(pred, gold)) / max(len(gold), 1)
    f1s = [prf([p == c for p in pred], [g == c for g in gold])["f1"] for c in DECISIONS
           if any(g == c for g in gold)]
    benign = [p for p, g in zip(pred, gold) if g == "allow"]
    risky = [p for p, g in zip(pred, gold) if g != "allow"]
    return {
        "accuracy": acc, "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        # how often a human gets asked, over everything
        "approval_frequency": sum(p == "ask" for p in pred) / max(len(pred), 1),
        # benign calls that still go through without a human
        "benign_completion": sum(p == "allow" for p in benign) / len(benign) if benign else None,
        # risky calls that went through unchecked
        "miss_rate": sum(p == "allow" for p in risky) / len(risky) if risky else None,
    }


def rate_allowed(pred: Sequence[str], types: Sequence[Sequence[str]], type_name: str) -> float | None:
    """E.g. destructive-action rate: share of samples of that type that were allowed."""
    sel = [p for p, ts in zip(pred, types) if type_name in ts]
    return sum(p == "allow" for p in sel) / len(sel) if sel else None
