"""Refit an adapter's temperatures on a calibration JSONL (design doc §10, E6).

    python -m training.calibrate --adapter weights/agent-precheck --calib data/agent_calib.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safejev.core.schema import read_jsonl

from .train_adapter import calibrate_meta, collect_logits


def calibrate(adapter_dir: str | Path, calib_path: str | Path, device: str | None = None) -> dict:
    from safejev.detectors.encoder import EncoderDetector
    det = EncoderDetector.load(adapter_dir, device=device, share_backbone=False)
    recs = list(read_jsonl(calib_path))
    zd, zt, zs = collect_logits(det.model, det.heads, det.tok, recs, det.meta, det.device)
    rep = calibrate_meta(det.meta, zd, zt, zs, recs)
    det.meta.save(Path(adapter_dir))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--device")
    a = ap.parse_args(argv)
    print(json.dumps(calibrate(a.adapter, a.calib, a.device), indent=2))


if __name__ == "__main__":
    main()
