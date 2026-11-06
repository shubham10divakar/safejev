"""E7 cost: latency percentiles and throughput of ``SafeJEV.scan_input``."""
from __future__ import annotations

import time

from ..core.detector import ScanInput


def bench(sj, inputs: list[ScanInput], repeat: int = 3, warmup: int = 3) -> dict:
    if not inputs:
        raise ValueError("bench needs at least one input")
    for i in range(min(warmup, len(inputs))):
        sj.scan_input(inputs[i], audit=False)
    lat = []
    t0 = time.perf_counter()
    for _ in range(repeat):
        for inp in inputs:
            s = time.perf_counter()
            sj.scan_input(inp, audit=False)
            lat.append((time.perf_counter() - s) * 1000)
    wall = time.perf_counter() - t0
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]  # noqa: E731
    return {"n": len(lat), "adapters": len(sj.detectors), "p50_ms": q(0.5), "p95_ms": q(0.95),
            "max_ms": lat[-1], "throughput_per_s": len(lat) / wall if wall else None}
