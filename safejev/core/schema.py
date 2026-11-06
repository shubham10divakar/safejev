"""Unified data record (design doc §6). Every training / eval sample becomes a ``Record``."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .taxonomy import DECISIONS, GRANULARITIES, MODULES, SEVERITIES, TYPE_TO_MODULE

SCHEMA_VERSION = "1"


@dataclass
class Source:
    dataset: str
    split: str
    version: str = ""
    license: str = ""


@dataclass
class RecordInput:
    user_goal: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    untrusted_context: list[str] = field(default_factory=list)
    proposed_action: dict[str, Any] | None = None
    tool_schema: dict[str, Any] | None = None
    text: str = ""          # for granularity "text" (a prompt / response / document)


@dataclass
class Label:
    decision: str
    types: list[str] = field(default_factory=list)
    severity: str = "none"


@dataclass
class Record:
    id: str
    source: Source
    granularity: str
    module: str
    input: RecordInput
    label: Label
    original_label: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- serialisation -------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Record":
        return cls(
            id=d["id"],
            source=Source(**d["source"]),
            granularity=d["granularity"],
            module=d["module"],
            input=RecordInput(**dict(d.get("input") or {})),
            label=Label(**d["label"]),
            original_label=dict(d.get("original_label") or {}),
            meta=dict(d.get("meta") or {}),
        )

    # ---- validation ----------------------------------------------------------------------
    def validate(self) -> "Record":
        errs = []
        if not self.id:
            errs.append("empty id")
        if self.module not in MODULES:
            errs.append(f"module {self.module!r} not in {MODULES}")
        if self.granularity not in GRANULARITIES:
            errs.append(f"granularity {self.granularity!r} not in {GRANULARITIES}")
        if self.label.decision not in DECISIONS:
            errs.append(f"decision {self.label.decision!r} not in {DECISIONS}")
        if self.label.severity not in SEVERITIES:
            errs.append(f"severity {self.label.severity!r} not in {SEVERITIES}")
        for t in self.label.types:
            if TYPE_TO_MODULE.get(t) != self.module:
                errs.append(f"type {t!r} does not belong to module {self.module!r}")
        if self.label.decision == "allow" and self.label.types:
            errs.append("decision 'allow' must have no types")
        if self.label.decision != "allow" and self.label.severity == "none":
            errs.append("non-allow decision needs a severity")
        if self.granularity == "step" and not self.input.proposed_action:
            errs.append("step records need input.proposed_action")
        if self.granularity == "text" and not self.input.text:
            errs.append("text records need input.text")
        if errs:
            raise ValueError(f"record {self.id!r}: " + "; ".join(errs))
        return self


def read_jsonl(path: str | Path) -> Iterator[Record]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Record.from_dict(json.loads(line))
            except Exception as e:  # noqa: BLE001 - re-raise with location
                raise ValueError(f"{path}:{n}: {e}") from e


def write_jsonl(records: Iterable[Record], path: str | Path, validate: bool = True) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            if validate:
                r.validate()
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
