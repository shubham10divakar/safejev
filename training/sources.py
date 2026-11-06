"""Source registry (design doc §9.1, §9.2). Every source is tagged before any training.

``use``:
- ``train-ok``  — may be used for training (licence permitting)
- ``train-split`` — only the named train split may be trained on; other splits are eval-only
- ``eval-only`` — never trained on (benchmark clean test sets, or the card forbids training)

``verified=False`` marks entries written from memory (the design doc's †): the identifier,
licence and terms must be checked by hand before the source is used. ``assert_trainable``
refuses unverified sources unless ``allow_unverified=True``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceInfo:
    name: str
    module: str
    use: str
    license: str = "unknown"
    verified: bool = False
    train_splits: tuple[str, ...] = ()
    notes: str = ""


SOURCES: dict[str, SourceInfo] = {s.name: s for s in [
    # guard
    SourceInfo("deepset-prompt-injections", "guard", "train-ok", notes="small; check licence"),
    SourceInfo("SPML", "guard", "train-ok"),
    SourceInfo("BIPIA", "guard", "train-ok", notes="indirect injection"),
    SourceInfo("jailbreak-in-the-wild", "guard", "train-ok", notes="dedupe heavily"),
    SourceInfo("synthetic-guard", "guard", "train-ok", "Apache-2.0", True,
               notes="own templates x carriers x obfuscations"),
    SourceInfo("JailbreakBench", "guard", "eval-only"),
    SourceInfo("AgentDojo", "guard", "eval-only", notes="agent loop: ASR + utility"),
    SourceInfo("InjecAgent", "guard", "eval-only"),
    # misuse
    SourceInfo("WildGuardMix", "misuse", "train-ok", notes="check licence; may be non-commercial"),
    SourceInfo("Aegis-2.0", "misuse", "train-ok", notes="check licence"),
    SourceInfo("phishing-email-corpus", "misuse", "train-ok"),
    SourceInfo("synthetic-misuse", "misuse", "train-ok", "Apache-2.0", True,
               notes="human-verify a test slice"),
    SourceInfo("HarmBench", "misuse", "eval-only"),
    SourceInfo("StrongREJECT", "misuse", "eval-only"),
    SourceInfo("CyberSecEval", "misuse", "eval-only"),
    SourceInfo("AgentHarm", "misuse", "eval-only", notes="dataset card asks not to train on it"),
    # agent
    SourceInfo("TS-Bench", "agent", "train-split", train_splits=("train",),
               notes="step-level; 'controversial' -> ask; eval split is eval-only"),
    SourceInfo("AuraGen", "agent", "train-ok", notes="Safiron recipe; synthesize + inject + filter"),
    SourceInfo("synthetic-mutations", "agent", "train-ok", "Apache-2.0", True,
               notes="training/mutations.py"),
    SourceInfo("Pre-Exec-Bench", "agent", "eval-only", notes="compare against Safiron"),
    SourceInfo("R-Judge", "agent", "eval-only", notes="eval by default; used in leave-one-out E2"),
    SourceInfo("HINTBench", "agent", "eval-only", notes="intrinsic-risk slice; leave-one-out E2"),
]}


class LeakageError(RuntimeError):
    pass


def is_trainable(dataset: str, split: str, allow_unverified: bool = False,
                 leave_one_out: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Whether (dataset, split) may be trained on. ``leave_one_out`` lets E2 train on R-Judge /
    HINTBench when they are *not* the held-out benchmark."""
    s = SOURCES.get(dataset)
    if s is None:
        return False, f"unknown source {dataset!r}: register it in training/sources.py first"
    if not s.verified and not allow_unverified:
        return False, f"{dataset} is not verified (licence / terms unchecked)"
    if s.use == "train-ok":
        return True, ""
    if s.use == "train-split":
        return (split in s.train_splits,
                "" if split in s.train_splits else f"{dataset}/{split} is eval-only")
    if leave_one_out and dataset in ("R-Judge", "HINTBench") and dataset not in leave_one_out:
        return True, ""
    return False, f"{dataset} is eval-only"


def assert_trainable(records, allow_unverified: bool = False, leave_one_out=()) -> None:
    bad: dict[str, str] = {}
    for r in records:
        ok, why = is_trainable(r.source.dataset, r.source.split, allow_unverified, tuple(leave_one_out))
        if not ok:
            bad.setdefault(f"{r.source.dataset}/{r.source.split}", why)
    if bad:
        raise LeakageError("refusing to train on: " + "; ".join(f"{k} ({v})" for k, v in bad.items()))
