"""Append-only JSONL audit log (design doc §12).

Each entry records the input hash (not the input itself, unless ``store_inputs=True``), the
adapters and versions that ran, the fired detections, the decision, the policy rule hit and the
approver.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from ..core.detector import ScanInput
from ..core.result import ScanResult


def input_hash(inp: ScanInput) -> str:
    blob = json.dumps(dataclasses.asdict(inp), sort_keys=True, ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | os.PathLike, store_inputs: bool = False):
        self.path = Path(path)
        self.store_inputs = store_inputs
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, inp: ScanInput, result: ScanResult, approval: Any = None,
               event: str = "scan", extra: dict | None = None) -> dict:
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z",
            "event": event,
            "input_hash": input_hash(inp),
            "tool": inp.tool_name or None,
            "adapters": result.adapters,
            "detections": [{"module": d.module, "type": d.type, "score": round(d.score, 4),
                            "severity": d.severity, "adapter": d.adapter} for d in result.detections],
            "risk_level": result.risk_level,
            "decision": result.decision,
            "policy_rule": result.policy_rule,
            "errors": result.errors,
            "latency_ms": round(result.latency_ms, 2),
        }
        if approval is not None:
            entry["approval"] = {"approved": approval.approved, "approver": approval.approver,
                                 "note": approval.note}
        if self.store_inputs:
            entry["input"] = dataclasses.asdict(inp)
        if extra:
            entry.update(extra)
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        return entry


def read_audit(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
