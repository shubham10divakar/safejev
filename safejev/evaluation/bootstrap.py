"""Bootstrap confidence intervals (design doc §15: small benchmarks need CIs)."""
from __future__ import annotations

import random
from typing import Callable, Sequence


def bootstrap_ci(metric: Callable[..., float | None], *columns: Sequence, n: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float | None, float | None, float | None]:
    """(point, low, high) for ``metric(*columns)`` with percentile bootstrap over rows."""
    point = metric(*columns)
    m = len(columns[0])
    if m == 0 or point is None:
        return point, None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        v = metric(*[[c[i] for i in idx] for c in columns])
        if v is not None:
            vals.append(v)
    if not vals:
        return point, None, None
    vals.sort()
    lo = vals[int(alpha / 2 * (len(vals) - 1))]
    hi = vals[int((1 - alpha / 2) * (len(vals) - 1))]
    return point, lo, hi
