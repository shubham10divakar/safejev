"""Detector interface and the engine's input type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .result import Detection


@dataclass
class ScanInput:
    """Everything a detector may look at. Every field is optional."""
    user_input: str = ""                                  # the user's prompt / goal
    response: str = ""                                    # model output to scan
    untrusted_context: list[str] = field(default_factory=list)   # retrieved docs, tool results
    agent_action: dict[str, Any] | None = None            # {"tool": ..., "arguments": {...}}
    history: list[dict[str, Any]] = field(default_factory=list)  # prior steps
    tool_schema: dict[str, Any] | None = None             # {"name", "params", "sensitive"}
    system_prompt: str = ""

    @property
    def tool_name(self) -> str:
        return str((self.agent_action or {}).get("tool") or "")

    @property
    def arguments(self) -> dict[str, Any]:
        args = (self.agent_action or {}).get("arguments") or {}
        return args if isinstance(args, dict) else {"_raw": args}

    @classmethod
    def from_record(cls, rec) -> "ScanInput":
        i = rec.input
        return cls(user_input=i.user_goal or i.text, untrusted_context=list(i.untrusted_context),
                   agent_action=i.proposed_action, history=list(i.history),
                   tool_schema=i.tool_schema)


@runtime_checkable
class Detector(Protocol):
    name: str          # adapter name, e.g. "agent-precheck"
    module: str        # guard / misuse / agent
    version: str

    def applies(self, inp: ScanInput) -> bool:
        """False when the input has nothing this detector can look at."""

    def detect(self, inp: ScanInput) -> list[Detection]:
        """One Detection per type scored (fired or not)."""


def adapter_id(det) -> str:
    return f"safejev/{det.name}@{det.version}"
