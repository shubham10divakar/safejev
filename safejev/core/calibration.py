"""Temperature scaling and calibration metrics (design doc §10, E6).

Pure Python, so the core does not need numpy; the trainer calls the same functions on lists.
"""
from __future__ import annotations

import math
from typing import Sequence


def softmax(z: Sequence[float], t: float = 1.0) -> list[float]:
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def nll(logits: Sequence[Sequence[float]], labels: Sequence[int], t: float) -> float:
    total = 0.0
    for z, y in zip(logits, labels):
        m = max(z)
        lse = m / t + math.log(sum(math.exp((x - m) / t) for x in z))
        total += lse - z[y] / t
    return total / max(len(labels), 1)


def fit_temperature(logits: Sequence[Sequence[float]], labels: Sequence[int],
                    lo: float = 0.05, hi: float = 20.0, iters: int = 80) -> float:
    """Minimise NLL over T by golden-section search on log T."""
    if not labels:
        return 1.0
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, labels, math.exp(c)), nll(logits, labels, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, labels, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, labels, math.exp(d))
    return math.exp((a + b) / 2)


def fit_binary_temperature(logits: Sequence[float], labels: Sequence[int], **kw) -> float:
    """Temperature for one-logit (sigmoid) heads: treated as 2-class logits [0, z]."""
    return fit_temperature([[0.0, z] for z in logits], list(labels), **kw)


def ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 15) -> float:
    """Binary ECE: probs are P(positive), labels 0/1. Equal-width bins."""
    n = len(probs)
    if n == 0:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    err = 0.0
    for b in bins:
        if b:
            conf = sum(p for p, _ in b) / len(b)
            acc = sum(y for _, y in b) / len(b)
            err += len(b) / n * abs(conf - acc)
    return err


def multiclass_ece(prob_rows: Sequence[Sequence[float]], labels: Sequence[int],
                   n_bins: int = 15) -> float:
    """Top-label ECE for multi-class probabilities."""
    conf = [max(r) for r in prob_rows]
    correct = [int(max(range(len(r)), key=r.__getitem__) == y) for r, y in zip(prob_rows, labels)]
    return ece(conf, correct, n_bins)


def reliability_bins(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10):
    """(bin_low, mean_conf, frac_positive, count) per non-empty bin, for reliability diagrams."""
    groups: dict[int, list[tuple[float, int]]] = {}
    for p, y in zip(probs, labels):
        groups.setdefault(min(int(p * n_bins), n_bins - 1), []).append((p, y))
    return [(i / n_bins, sum(p for p, _ in g) / len(g), sum(y for _, y in g) / len(g), len(g))
            for i, g in sorted(groups.items())]
