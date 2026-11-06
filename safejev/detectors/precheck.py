"""Heuristic ``agent-precheck``: a rule baseline for the pre-execution tool-call check.

It scores every agent type in the taxonomy from the proposed call, the user goal, the history,
untrusted context and the tool schema. The rules mirror the mutation generator's edit types
(design doc §9.4), so the baseline and the trained adapter are compared on the same ground,
including the hard negatives: a destructive or external call the user explicitly asked for.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.detector import ScanInput, adapter_id
from ..core.result import Detection
from ..core.taxonomy import DEFAULT_SEVERITY, MODULE_TYPES
from .heuristic import HEURISTIC_VERSION, injection_hits, misuse_hits, noisy_or

_I = re.IGNORECASE

DESTRUCTIVE_TOOL = re.compile(r"(delete|remove|drop|truncate|wipe|erase|destroy|purge|format|"
                              r"kill|terminate|revoke|uninstall|reset|overwrite|\brm\b)", _I)
EXTERNAL_TOOL = re.compile(r"(send|email|mail|post|upload|share|publish|webhook|http|request|"
                           r"fetch_url|curl|transfer|pay|wire|message|tweet|slack|sms)", _I)
WRITE_TOOL = re.compile(r"(write|update|create|modify|edit|set|move|rename|insert|commit|push|"
                        r"deploy|install|execute|run|exec|shell|bash|transfer|pay|send|post)", _I)
FINANCE_TOOL = re.compile(r"(transfer|pay|payment|wire|withdraw|refund|purchase|buy|order|"
                          r"invoice|transaction)", _I)
SHELL_TOOL = re.compile(r"(shell|bash|terminal|exec|run_command|cmd|powershell|subprocess)", _I)

READ_ONLY_GOAL = re.compile(r"^\s*(please\s+)?(can you\s+)?(find|list|show|search|look up|"
                            r"lookup|read|check|summari[sz]e|count|get|tell me|what|which|how "
                            r"many|display|view|review|explain|compare|analy[sz]e|identify|"
                            r"locate|report)\b", _I)
GOAL_VERB = {
    "delete": re.compile(r"\b(delete|remove|erase|wipe|clean ?up|get rid of|purge|drop|"
                         r"clear)\b", _I),
    "send": re.compile(r"\b(send|email|mail|forward|share|post|reply|message|notify|text)\b", _I),
    "pay": re.compile(r"\b(pay|transfer|send money|wire|refund|purchase|buy|order)\b", _I),
    "write": re.compile(r"\b(write|update|create|change|edit|modify|save|rename|move|add|set|"
                        r"install|deploy|run|execute)\b", _I),
}

DANGEROUS_SHELL = [
    (re.compile(r"\brm\s+-[a-z]*r[a-z]*f?\s+(/|~|\*|\$HOME)", _I), "recursive delete of a root path"),
    (re.compile(r"(curl|wget)[^|;]*\|\s*(ba|z)?sh\b", _I), "pipes a download into a shell"),
    (re.compile(r"\b(nc|ncat|netcat)\b.*\s-e\s", _I), "netcat with -e (reverse shell)"),
    (re.compile(r"/dev/tcp/", _I), "bash /dev/tcp reverse shell"),
    (re.compile(r"\bmkfs\b|\bdd\s+if=.*of=/dev/", _I), "formats or overwrites a disk"),
    (re.compile(r"chmod\s+(-R\s+)?777", _I), "world-writable permissions"),
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"), "fork bomb"),
    (re.compile(r"\b(shutdown|reboot|halt)\b", _I), "shuts the machine down"),
]
DANGEROUS_SQL = re.compile(r"\b(drop\s+(table|database)|truncate\s+table|delete\s+from\s+\w+"
                           r"\s*(;|$)|update\s+\w+\s+set\s+[^;]*?(;|$)(?!.*where))", _I)

SECRET = re.compile(r"(-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(sk|pk|rk)-[A-Za-z0-9_-]{16,}|"
                    r"\bAKIA[0-9A-Z]{16}\b|\bghp_[A-Za-z0-9]{30,}|\bpassword\s*[:=]|\bapi[_-]?key"
                    r"\s*[:=]|\b\d{3}-\d{2}-\d{4}\b|\b(?:\d[ -]?){13,16}\b)", _I)
SENSITIVE_PATH = re.compile(r"(\.env\b|id_rsa|\.ssh/|\.aws/credentials|/etc/(passwd|shadow)|"
                            r"passwords?\.(txt|csv|json)|secrets?\.(ya?ml|json|txt)|\.pem\b|"
                            r"keychain|wallet)", _I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_HOST = re.compile(r"https?://([\w.-]+)", _I)
PATH = re.compile(r"(?<![\w.])(/[\w.\-/]*|[A-Za-z]:\\[\w.\-\\ ]*|~/[\w.\-/]*)")
NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w.])")
WILDCARD_ARG = re.compile(r"^(\*|%|all|any|everything|\.\*|/|~|\*\.\*|\.)$", _I)
BOOL_RISK_KEYS = ("recursive", "force", "all", "overwrite", "no_confirm", "skip_confirmation",
                  "sudo", "as_admin", "public", "permanent", "hard")


def _flatten(x: Any, key: str = "") -> list[tuple[str, Any]]:
    if isinstance(x, dict):
        return [kv for k, v in x.items() for kv in _flatten(v, f"{key}.{k}" if key else str(k))]
    if isinstance(x, (list, tuple)):
        return [kv for v in x for kv in _flatten(v, key)]
    return [(key, x)]


def _num(s: str) -> float | None:
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


@dataclass
class _Acc:
    types: tuple[str, ...]
    hits: dict[str, list[tuple[float, str]]] = field(default_factory=dict)

    def add(self, t: str, w: float, reason: str):
        self.hits.setdefault(t, []).append((w, reason))

    def damp(self, t: str, factor: float, reason: str):
        if t in self.hits:
            self.hits[t] = [(w * factor, r) for w, r in self.hits[t]] + [(0.0, reason)]


class PrecheckDetector:
    """Rule baseline for ``agent-precheck``. Returns one Detection per agent type."""

    name = "agent-precheck"
    module = "agent"
    version = HEURISTIC_VERSION

    def __init__(self, threshold: float = 0.5, trusted_domains: tuple[str, ...] = ()):
        self.threshold = threshold
        self.trusted_domains = tuple(d.lower() for d in trusted_domains)
        self.types = MODULE_TYPES["agent"]

    def applies(self, inp: ScanInput) -> bool:
        return bool(inp.agent_action)

    # ---------------------------------------------------------------------------------------
    def detect(self, inp: ScanInput) -> list[Detection]:
        acc = _Acc(self.types)
        tool = inp.tool_name
        args = inp.arguments
        flat = _flatten(args)
        arg_text = json.dumps(args, ensure_ascii=False, default=str)
        goal = inp.user_input or ""
        known = _norm(" ".join([goal] + [str(h.get("content", "")) for h in inp.history
                                         if h.get("role") in ("user", "tool_result", "tool")]))
        untrusted = list(inp.untrusted_context) + [
            str(h.get("content", "")) for h in inp.history
            if h.get("role") in ("tool_result", "tool", "observation", "retrieved")]
        tool_l = tool.lower().replace("-", "_")

        self._malformed(inp, tool, args, acc)
        self._malicious_request(goal, acc)
        self._harmful(tool_l, flat, arg_text, acc)
        self._destructive(tool_l, arg_text, acc)
        self._risky_args(tool_l, flat, goal, acc)
        self._intent(tool_l, flat, goal, known, acc)
        self._exfil(tool_l, flat, arg_text, goal, inp, acc)
        self._injected(untrusted, flat, goal, acc)
        self._unauthorized(tool_l, inp, goal, acc)
        self._multi_step(tool_l, inp, acc)
        self._hard_negative(tool_l, flat, goal, acc)

        out = []
        for t in self.types:
            hs = acc.hits.get(t, [])
            score = noisy_or(w for w, _ in hs)
            reasons = "; ".join(r for w, r in hs if r)
            out.append(Detection(self.module, t, score, adapter_id(self), DEFAULT_SEVERITY[t],
                                 self.threshold, reasons))
        return out

    # ---- individual rule groups -----------------------------------------------------------
    @staticmethod
    def _malformed(inp: ScanInput, tool: str, args: dict, acc: _Acc):
        if not tool:
            acc.add("malformed_call", 0.9, "no tool name")
        if "_raw" in args:
            acc.add("malformed_call", 0.7, "arguments are not an object")
        schema = inp.tool_schema
        if not schema:
            return
        if schema.get("name") and tool and schema["name"] != tool:
            acc.add("malformed_call", 0.8, f"tool {tool!r} does not match schema {schema['name']!r}")
        params = schema.get("params") or schema.get("parameters") or {}
        if isinstance(params, dict) and "properties" in params:      # JSON-schema style
            required = params.get("required", [])
            params = params["properties"]
        else:
            required = schema.get("required", [])
        if isinstance(params, dict) and params:
            for k in args:
                if k != "_raw" and k not in params:
                    acc.add("malformed_call", 0.75, f"invented parameter {k!r}")
            for k in required:
                if k not in args:
                    acc.add("malformed_call", 0.7, f"missing required parameter {k!r}")
            types = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
                     "array": list, "object": dict}
            for k, v in args.items():
                spec = params.get(k)
                want = spec.get("type") if isinstance(spec, dict) else spec
                if isinstance(want, str) and want in types and not isinstance(v, types[want]):
                    acc.add("malformed_call", 0.6, f"parameter {k!r} should be {want}")

    @staticmethod
    def _malicious_request(goal: str, acc: _Acc):
        for r in misuse_hits(goal):
            acc.add("malicious_request", r.weight, f"goal: {r.reason}")

    @staticmethod
    def _harmful(tool_l: str, flat, arg_text: str, acc: _Acc):
        if SHELL_TOOL.search(tool_l) or any(k.endswith(("command", "cmd", "script")) for k, _ in flat):
            for pat, why in DANGEROUS_SHELL:
                if pat.search(arg_text):
                    acc.add("harmful_tool", 0.85, f"shell: {why}")
        if DANGEROUS_SQL.search(arg_text):
            acc.add("destructive_action", 0.75, "destructive SQL statement")

    @staticmethod
    def _destructive(tool_l: str, arg_text: str, acc: _Acc):
        if DESTRUCTIVE_TOOL.search(tool_l):
            acc.add("destructive_action", 0.6, f"tool {tool_l!r} is destructive")

    @staticmethod
    def _risky_args(tool_l: str, flat, goal: str, acc: _Acc):
        for k, v in flat:
            key = k.split(".")[-1].lower()
            if isinstance(v, str) and WILDCARD_ARG.match(v.strip()):
                acc.add("risky_arguments", 0.65, f"{k}={v!r} widens scope to everything")
            if isinstance(v, bool) and v and key in BOOL_RISK_KEYS:
                acc.add("risky_arguments", 0.45, f"{k}=true")
            if isinstance(v, str) and SENSITIVE_PATH.search(v):
                acc.add("risky_arguments", 0.6, f"{k} touches a sensitive file")
            if (isinstance(v, (int, float)) and not isinstance(v, bool) and FINANCE_TOOL.search(tool_l)
                    and key in ("amount", "value", "quantity", "qty", "total") and v >= 10_000):
                acc.add("risky_arguments", 0.5, f"{k}={v} is a large amount")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and key in ("limit", "count", "max_results") and v >= 10_000:
                acc.add("risky_arguments", 0.35, f"{k}={v} is very large")

    @staticmethod
    def _intent(tool_l: str, flat, goal: str, known: str, acc: _Acc):
        if not goal:
            return
        g = goal.lower()
        # read-only goal, side-effecting action
        if READ_ONLY_GOAL.search(goal) and not any(p.search(goal) for p in GOAL_VERB.values()):
            if DESTRUCTIVE_TOOL.search(tool_l):
                acc.add("intent_mismatch", 0.85, "goal is read-only; the call deletes")
            elif EXTERNAL_TOOL.search(tool_l) or WRITE_TOOL.search(tool_l):
                acc.add("intent_mismatch", 0.65, "goal is read-only; the call has side effects")
        # values in the call the user never mentioned: recipients, hosts, paths, amounts
        for k, v in flat:
            if not isinstance(v, (str, int, float)) or isinstance(v, bool):
                continue
            sv = str(v)
            for em in EMAIL.findall(sv):
                if em.lower() not in known:
                    acc.add("intent_mismatch", 0.55, f"recipient {em} not in the goal or history")
            if FINANCE_TOOL.search(tool_l) and k.split(".")[-1].lower() in ("amount", "value", "total"):
                goal_nums = {_num(n) for n in NUMBER.findall(g)} - {None}
                val = _num(sv)
                if goal_nums and val is not None and val not in goal_nums:
                    acc.add("intent_mismatch", 0.7, f"amount {sv} differs from the goal")
        goal_paths = [p.rstrip("/\\").lower() for p in PATH.findall(goal) if len(p) > 1]
        if goal_paths:
            for k, v in flat:
                if isinstance(v, str):
                    for p in PATH.findall(v):
                        pl = p.lower()
                        if len(p) > 1 and not any(pl.startswith(gp) for gp in goal_paths):
                            acc.add("intent_mismatch", 0.5, f"path {p} is outside {goal_paths[0]}")

    def _exfil(self, tool_l: str, flat, arg_text: str, goal: str, inp: ScanInput, acc: _Acc):
        if not EXTERNAL_TOOL.search(tool_l) and not any(URL_HOST.search(str(v)) for _, v in flat):
            return
        g = goal.lower()
        hosts = {h.lower() for _, v in flat for h in URL_HOST.findall(str(v))}
        emails = {e.lower() for _, v in flat for e in EMAIL.findall(str(v))}
        unknown = [h for h in hosts if h not in g and not h.endswith(self.trusted_domains or ("\0",))]
        unknown += [e for e in emails if e not in g and
                    not e.split("@")[-1].endswith(self.trusted_domains or ("\0",))]
        carries_secret = bool(SECRET.search(arg_text)) or bool(SENSITIVE_PATH.search(arg_text))
        if carries_secret:
            acc.add("data_exfiltration", 0.7 if unknown else 0.45,
                    "call sends secret-looking data" + (f" to {unknown[0]}" if unknown else ""))
        if unknown:
            hist_secret = any(SECRET.search(str(h.get("content", ""))) or
                              SENSITIVE_PATH.search(str(h.get("content", ""))) for h in inp.history)
            if hist_secret:
                acc.add("data_exfiltration", 0.6, "earlier steps read secrets; now sending out")

    @staticmethod
    def _injected(untrusted: list[str], flat, goal: str, acc: _Acc):
        g = goal.lower()
        arg_vals = [str(v).lower() for _, v in flat if isinstance(v, (str, int, float))]
        for text in untrusted:
            hits = injection_hits(text)
            if not hits:
                continue
            tl = text.lower()
            planted = {e.lower() for e in EMAIL.findall(text)} | {h.lower() for h in URL_HOST.findall(text)}
            echoed = [p for p in planted if p not in g and any(p in a for a in arg_vals)]
            if echoed:
                acc.add("injected_goal", 0.9, f"call uses {echoed[0]} planted by injected text")
            elif any(len(a) > 6 and a in tl and a not in g for a in arg_vals):
                acc.add("injected_goal", 0.6, "call follows values from injected text")
            else:
                acc.add("injected_goal", 0.3, "untrusted context contains instructions")

    @staticmethod
    def _unauthorized(tool_l: str, inp: ScanInput, goal: str, acc: _Acc):
        sensitive = bool((inp.tool_schema or {}).get("sensitive"))
        if not sensitive or not goal:
            return
        family = [v for v, p in GOAL_VERB.items() if p.search(goal)]
        tool_family = ("delete" if DESTRUCTIVE_TOOL.search(tool_l) else
                       "pay" if FINANCE_TOOL.search(tool_l) else
                       "send" if EXTERNAL_TOOL.search(tool_l) else "write")
        if tool_family not in family:
            acc.add("unauthorized_action", 0.6, f"sensitive {tool_family} tool not requested by the user")

    @staticmethod
    def _multi_step(tool_l: str, inp: ScanInput, acc: _Acc):
        calls = [h for h in inp.history if h.get("role") == "tool_call"]
        if len(calls) >= 3:
            last = [json.dumps(c.get("content"), sort_keys=True, default=str) for c in calls[-3:]]
            if len(set(last)) == 1:
                acc.add("multi_step_risk", 0.55, "same call repeated 3 times (loop)")
        read_secret = any(SENSITIVE_PATH.search(json.dumps(c.get("content"), default=str))
                          for c in calls)
        if read_secret and EXTERNAL_TOOL.search(tool_l):
            acc.add("multi_step_risk", 0.6, "read a sensitive file earlier, now an external call")
        destructive_before = sum(bool(DESTRUCTIVE_TOOL.search(json.dumps(c.get("content"), default=str)))
                                 for c in calls)
        if destructive_before >= 2 and DESTRUCTIVE_TOOL.search(tool_l):
            acc.add("multi_step_risk", 0.45, "a run of destructive calls")

    @staticmethod
    def _hard_negative(tool_l: str, flat, goal: str, acc: _Acc):
        """The user explicitly asked for exactly this: damp destructive / scope / mismatch."""
        if not goal:
            return
        g = goal.lower()
        asked_delete = bool(GOAL_VERB["delete"].search(goal)) and DESTRUCTIVE_TOOL.search(tool_l)
        asked_send = bool(GOAL_VERB["send"].search(goal) or GOAL_VERB["pay"].search(goal)) and \
            EXTERNAL_TOOL.search(tool_l)
        targets = [str(v).lower() for _, v in flat if isinstance(v, str) and len(str(v)) > 1]
        named = bool(targets) and all(t in g or any(part in g for part in re.split(r"[/\\]", t)
                                                    if len(part) > 3) for t in targets)
        goal_nums = {_num(n) for n in NUMBER.findall(g)}
        nums = [v for _, v in flat if isinstance(v, (int, float)) and not isinstance(v, bool)]
        named = named and all(float(v) in goal_nums for v in nums)
        if (asked_delete or asked_send) and named:
            for t in ("destructive_action", "intent_mismatch", "risky_arguments",
                      "unauthorized_action"):
                acc.damp(t, 0.25, "")
            acc.damp("data_exfiltration", 0.5, "")
        elif asked_delete:
            acc.damp("destructive_action", 0.6, "")
