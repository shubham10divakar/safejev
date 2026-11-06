"""Train one adapter (design doc §10).

    python -m training.train_adapter --name agent-precheck --module agent \
        --train data/agent_train.jsonl --calib data/agent_calib.jsonl \
        --backbone answerdotai/ModernBERT-base --out weights/agent-precheck

Stage 1 trains one adapter per capability (or per source, ``--tier research``); stage 2 joint
multi-task is the same command over a mixed JSONL. ``--full`` fine-tunes the whole backbone
(E1 baseline) instead of LoRA.

Loss = CE(decision, class-balanced) + BCE(types, pos-weighted) + 0.5 * CE(severity).
After training, temperatures are fitted per head on the calibration split, and ECE before /
after goes into ``train_report.json`` (E6).

``--dry-run`` generates a few mutation records and trains a handful of steps; it is the
smoke test for a new machine (still needs the backbone locally or on the Hub).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from safejev.core.calibration import (ece, fit_binary_temperature, fit_temperature,
                                      multiclass_ece, sigmoid, softmax)
from safejev.core.detector import ScanInput
from safejev.core.schema import Record, read_jsonl
from safejev.core.taxonomy import DECISIONS, SEVERITIES
from safejev.detectors.text import render_pair

from .leakage import assert_no_canaries
from .sources import assert_trainable


def _targets(recs: list[Record], types: list[str]):
    d = [DECISIONS.index(r.label.decision) for r in recs]
    t = [[1.0 if ty in r.label.types else 0.0 for ty in types] for r in recs]
    s = [SEVERITIES.index(r.label.severity) for r in recs]
    return d, t, s


def _class_weights(labels: list[int], n: int) -> list[float]:
    counts = [labels.count(i) for i in range(n)]
    tot = sum(counts)
    return [tot / (n * c) if c else 0.0 for c in counts]


def _batches(recs, bs, rng, shuffle=True):
    idx = list(range(len(recs)))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), bs):
        yield [recs[i] for i in idx[s:s + bs]]


def _encode(tok, recs, meta, device):
    pairs = [render_pair(ScanInput.from_record(r) if r.granularity != "text"
                         else ScanInput(user_input=r.input.text, untrusted_context=r.input.untrusted_context),
                         meta.granularity, meta.last_k) for r in recs]
    enc = tok([p[0] or " " for p in pairs], [p[1] or " " for p in pairs], truncation="longest_first",
              max_length=meta.max_length, padding=True, return_tensors="pt")
    return enc.to(device)


def collect_logits(model, heads, tok, recs, meta, device, bs=32):
    import torch
    from safejev.detectors.encoder import pool
    model.eval(), heads.eval()
    zd, zt, zs = [], [], []
    with torch.inference_mode():
        for b in _batches(recs, bs, None, shuffle=False):
            enc = _encode(tok, b, meta, device)
            d, t, s = heads(pool(model(**enc), enc["attention_mask"]))
            zd += d.float().cpu().tolist()
            zt += t.float().cpu().tolist()
            zs += s.float().cpu().tolist()
    return zd, zt, zs


def calibrate_meta(meta, zd, zt, zs, recs) -> dict:
    """Fit temperatures on a calibration split; returns ECE before / after."""
    d, t, s = _targets(recs, meta.types)
    rep = {"n": len(recs)}
    rep["ece_decision_before"] = multiclass_ece([softmax(z) for z in zd], d)
    meta.t_decision = fit_temperature(zd, d)
    rep["ece_decision_after"] = multiclass_ece([softmax(z, meta.t_decision) for z in zd], d)
    meta.t_severity = fit_temperature(zs, s)
    tts, before, after = [], [], []
    for j in range(len(meta.types)):
        col = [row[j] for row in zt]
        lab = [int(row[j]) for row in t]
        tt = fit_binary_temperature(col, lab) if any(lab) and not all(lab) else 1.0
        tts.append(tt)
        before.append(ece([sigmoid(z) for z in col], lab))
        after.append(ece([sigmoid(z / tt) for z in col], lab))
    meta.t_types = tts
    rep["ece_types_before"] = sum(before) / len(before)
    rep["ece_types_after"] = sum(after) / len(after)
    meta.calibrated = True
    return rep


def train(train_recs: list[Record], meta, out_dir: str | Path, calib_recs: list[Record] | None = None,
          epochs: int = 3, lr: float = 2e-4, head_lr: float = 1e-3, batch_size: int = 16,
          max_steps: int | None = None, device: str | None = None, seed: int = 0,
          backbone_path: str | None = None, allow_unverified: bool = False,
          leave_one_out: tuple[str, ...] = (), log_every: int = 50) -> dict:
    import torch
    from torch import nn

    from safejev.detectors import encoder

    assert_trainable(train_recs, allow_unverified, leave_one_out)
    assert_no_canaries(train_recs)
    for r in train_recs:
        r.validate()
        if r.module != meta.module:
            raise ValueError(f"record {r.id} is module {r.module!r}, adapter is {meta.module!r}")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = encoder.default_device(device)
    model, heads, tok = encoder.build(meta, device, backbone_path)

    d_all, t_all, _ = _targets(train_recs, meta.types)
    w_dec = torch.tensor(_class_weights(d_all, len(DECISIONS)), device=device)
    pos = [sum(row[j] for row in t_all) for j in range(len(meta.types))]
    n = len(train_recs)
    pos_w = torch.tensor([min((n - p) / p, 50.0) if p else 1.0 for p in pos], device=device)
    ce_dec = nn.CrossEntropyLoss(weight=w_dec)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ce_sev = nn.CrossEntropyLoss()

    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr},
              {"params": heads.parameters(), "lr": head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps_per_epoch = math.ceil(n / batch_size)
    total = min(max_steps or 10**12, epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, total // 10)) * max(0.0, 1 - s / total))

    step, t0, history = 0, time.time(), []
    model.train(), heads.train()
    for ep in range(epochs):
        for b in _batches(train_recs, batch_size, rng):
            enc = _encode(tok, b, meta, device)
            d, t, s = _targets(b, meta.types)
            zd, zt, zs = heads(encoder.pool(model(**enc), enc["attention_mask"]))
            loss = (ce_dec(zd, torch.tensor(d, device=device))
                    + bce(zt, torch.tensor(t, device=device))
                    + 0.5 * ce_sev(zs, torch.tensor(s, device=device)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
            opt.step()
            sched.step()
            step += 1
            history.append(float(loss))
            if log_every and step % log_every == 0:
                print(f"step {step}/{total} loss {sum(history[-log_every:]) / log_every:.4f}", flush=True)
            if step >= total:
                break
        if step >= total:
            break

    report = {"steps": step, "epochs_run": ep + 1, "n_train": n, "seconds": round(time.time() - t0, 1),
              "loss_first": history[0] if history else None, "loss_last": history[-1] if history else None,
              "sources": sorted({f"{r.source.dataset}/{r.source.split}" for r in train_recs})}
    meta.training_sources = report["sources"]
    if calib_recs:
        zd, zt, zs = collect_logits(model, heads, tok, calib_recs, meta, device)
        report["calibration"] = calibrate_meta(meta, zd, zt, zs, calib_recs)
    out = encoder.save(out_dir, meta, model, heads, tok)
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    from safejev.core.registry import ADAPTERS
    from safejev.detectors.encoder import AdapterMeta

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="agent-precheck")
    ap.add_argument("--module", default=None)
    ap.add_argument("--train")
    ap.add_argument("--calib")
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--last-k", type=int, default=None, help="history steps rendered (E4)")
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--leave-one-out", nargs="*", default=())
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    info = ADAPTERS.get(a.name)
    module = a.module or (info.module if info else None)
    if module is None:
        ap.error("--module is required for an unregistered adapter name")
    meta = AdapterMeta(name=a.name, module=module, backbone=a.backbone, version=a.version,
                       granularity=info.granularity if info else "step", max_length=a.max_length,
                       last_k=a.last_k, lora=not a.full, lora_r=a.lora_r, lora_alpha=2 * a.lora_r,
                       tier=info.tier if info else "research")
    if a.dry_run:
        from .mutations import builtin_seeds, generate
        recs = list(generate(builtin_seeds(random.Random(a.seed), 2), random.Random(a.seed)))
        rep = train(recs, meta, a.out, recs[:16], epochs=1, max_steps=a.max_steps or 3,
                    batch_size=4, device=a.device, seed=a.seed)
    else:
        if not a.train:
            ap.error("--train is required (or use --dry-run)")
        recs = list(read_jsonl(a.train))
        calib = list(read_jsonl(a.calib)) if a.calib else None
        rep = train(recs, meta, a.out, calib, a.epochs, a.lr, batch_size=a.batch_size,
                    max_steps=a.max_steps, device=a.device, seed=a.seed,
                    allow_unverified=a.allow_unverified, leave_one_out=tuple(a.leave_one_out))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
