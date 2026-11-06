"""Heuristic text detectors: guard-injection, guard-jailbreak, misuse-request.

These are deterministic baselines. They need no weights, so ``pip install safejev`` works out
of the box, and every experiment has a rule baseline to beat. Scores are a noisy-OR of rule
weights; they are **not** calibrated probabilities, which the adapter version marks with
``+heuristic``.
"""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field

from ..core.detector import ScanInput, adapter_id
from ..core.result import Detection
from ..core.taxonomy import DEFAULT_SEVERITY

HEURISTIC_VERSION = "0.1.0+heuristic"
_F = re.IGNORECASE | re.DOTALL


@dataclass
class Rule:
    type: str
    pattern: re.Pattern
    weight: float
    reason: str


def _r(type_: str, pattern: str, weight: float, reason: str) -> Rule:
    return Rule(type_, re.compile(pattern, _F), weight, reason)


def noisy_or(weights) -> float:
    p = 1.0
    for w in weights:
        p *= 1.0 - w
    return 1.0 - p


# --------------------------------------------------------------------------------------------
# pattern sets
# --------------------------------------------------------------------------------------------
_IGNORE = (r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}\b(previous|prior|above|"
           r"earlier|all|any|your|the|system|original)\b.{0,30}\b(instructions?|prompts?|rules?|"
           r"directives?|guidelines|context)\b")

INJECTION_RULES = [
    _r("direct_injection", _IGNORE, 0.85, "asks to ignore earlier instructions"),
    _r("direct_injection", r"\b(new|updated|real|actual)\s+(instructions?|task|objective)\s*[:\-]",
       0.6, "announces replacement instructions"),
    _r("direct_injection", r"\b(reveal|print|show|repeat|output|leak)\b.{0,30}\b(system|hidden|"
       r"initial|secret)\s+(prompt|instructions?|message)", 0.75, "asks for the system prompt"),
    _r("context_manipulation", r"(<\s*/?\s*(system|assistant|im_start|im_end)\s*>|\[/?INST\]|"
       r"<\|(system|im_start|endoftext)\|>|^\s*#{2,}\s*(system|instruction)s?\b)", 0.7,
       "injects chat-template or role markers"),
    _r("context_manipulation", r"\b(end of (the )?(document|context|user input)|the (conversation|"
       r"above) (is|was) (over|a test))\b", 0.45, "tries to close the current context"),
    _r("hierarchy_violation", r"\b(you are now|from now on,? you|act as (an? )?(unrestricted|"
       r"admin|root|developer))\b", 0.5, "tries to reassign the assistant's role"),
    _r("hierarchy_violation", r"\b(developer|admin|god|sudo|maintenance) mode\b", 0.6,
       "claims a privileged mode"),
]

# Instructions addressed to the model inside *untrusted* content (retrieved docs, tool results)
INDIRECT_RULES = [
    _r("indirect_injection", _IGNORE, 0.9, "untrusted content tells the model to ignore instructions"),
    _r("indirect_injection", r"\b(ai|assistant|chatbot|llm|agent|model)\b.{0,20}\b(must|should|"
       r"please|now)\b.{0,60}\b(send|forward|email|upload|post|transfer|delete|run|execute|"
       r"click|visit|call)\b", 0.75, "untrusted content gives the agent an action"),
    _r("indirect_injection", r"\b(when you (read|see|process) this|if you are an? (ai|llm|"
       r"assistant|agent))\b", 0.7, "untrusted content addresses the model directly"),
    _r("indirect_injection", r"\b(do not|don't|never) (tell|inform|mention|alert)\b.{0,20}\b(the )?"
       r"user\b", 0.7, "untrusted content asks to hide something from the user"),
]

JAILBREAK_RULES = [
    _r("jailbreak", r"\b(DAN|do anything now|STAN|DUDE|AIM)\b.{0,80}\b(mode|prompt|persona|"
       r"anything)\b", 0.85, "known jailbreak persona"),
    _r("jailbreak", r"\bjail\s?break(ing|ed)?\b", 0.5, "mentions jailbreaking"),
    _r("jailbreak", r"\b(no|without( any)?|free (of|from)|ignore( all)?)\s+(ethical|moral|safety|"
       r"content)?\s*(restrictions|limitations|filters|guidelines|rules|censorship)\b", 0.7,
       "asks to drop safety constraints"),
    _r("jailbreak", r"\b(pretend|imagine|roleplay|role-play|act)\b.{0,40}\b(no (rules|limits|"
       r"filters)|unfiltered|uncensored|evil|amoral)\b", 0.8, "role-play to escape constraints"),
    _r("jailbreak", r"\b(hypothetically|for a (novel|story|movie)|purely fictional)\b.{0,80}"
       r"\b(how to|step[- ]by[- ]step|instructions for)\b", 0.45, "fictional framing for instructions"),
    _r("jailbreak", r"\bstay in character\b|\byou (cannot|can't|must not) refuse\b", 0.6,
       "forbids refusal"),
    _r("jailbreak", r"\b(my (late )?grandma|grandmother)\b.{0,80}\b(used to|would)\b", 0.35,
       "sentimental framing (grandma exploit)"),
]

MISUSE_RULES = [
    _r("cyber_abuse", r"\b(write|create|build|code|generate|make)\b.{0,40}\b(ransomware|keylogger|"
       r"malware|virus|worm|trojan|rootkit|botnet|info-?stealer|backdoor)\b", 0.85,
       "asks to build malware"),
    _r("cyber_abuse", r"\b(exploit|0-?day|zero-day|rce)\b.{0,40}\b(for|against|on)\b.{0,40}"
       r"\b(cve-\d{4}-\d+|server|website|router|bank)\b", 0.6, "asks for a working exploit"),
    _r("cyber_abuse", r"\b(bypass|evade|disable)\b.{0,30}\b(antivirus|av|edr|defender|firewall|"
       r"detection|2fa|mfa)\b", 0.7, "asks to evade security controls"),
    _r("cyber_abuse", r"\b(ddos|dos attack|flood)\b.{0,40}\b(site|server|ip|target)\b", 0.7,
       "asks for a denial-of-service attack"),
    _r("cyber_abuse", r"\b(steal|harvest|dump|exfiltrate|crack)\b.{0,30}\b(passwords?|credentials|"
       r"cookies|tokens|hashes|credit cards?)\b", 0.75, "asks to steal credentials"),
    _r("phishing_fraud", r"\bphishing\b.{0,30}\b(email|page|site|message|sms|kit|campaign)\b|"
       r"\b(email|page|site|message|sms)\b.{0,30}\bphishing\b", 0.85, "asks for phishing material"),
    _r("phishing_fraud", r"\b(pretend|pose|posing|impersonat\w*)\b.{0,30}\b(bank|paypal|irs|"
       r"microsoft|apple|amazon|support team|it department)\b", 0.7,
       "impersonates a trusted organisation"),
    _r("phishing_fraud", r"\b(fake|forged|counterfeit)\b.{0,20}\b(invoice|receipt|check|cheque|"
       r"id|passport|bank statement)\b", 0.75, "asks for forged documents"),
    _r("phishing_fraud", r"\b(urgent(ly)?|immediately|within 24 hours)\b.{0,60}\b(verify|confirm|"
       r"update)\b.{0,30}\b(account|password|payment|card)\b", 0.5, "urgent credential lure"),
    _r("deceptive_content", r"\b(fake|false|fabricated|misleading)\b.{0,20}\b(news|article|"
       r"reviews?|testimonials?|quotes?|study|statistics)\b", 0.7, "asks for fabricated content"),
    _r("deceptive_content", r"\b(disinformation|propaganda|astroturf\w*|sockpuppet\w*)\b", 0.65,
       "asks for a disinformation campaign"),
    _r("deceptive_content", r"\b(deepfake|voice clone)\b", 0.5, "asks for synthetic impersonation"),
    _r("general_harm", r"\b(make|build|synthesi[sz]e|cook|produce)\b.{0,30}\b(bomb|explosive|"
       r"nerve agent|sarin|ricin|meth(amphetamine)?|bioweapon|pipe bomb)\b", 0.9,
       "asks for weapons or drug synthesis"),
]

_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_B64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/=])")


def _decoded_base64(text: str) -> list[str]:
    out = []
    for m in _B64.finditer(text):
        try:
            s = base64.b64decode(m.group(0) + "=" * (-len(m.group(0)) % 4)).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if sum(ch.isprintable() for ch in s) > 0.9 * len(s):
            out.append(s)
    return out


# --------------------------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------------------------
@dataclass
class _TextDetector:
    name: str
    module: str
    types: tuple[str, ...]
    threshold: float = 0.5
    version: str = HEURISTIC_VERSION
    _rules: list = field(default_factory=list, repr=False)

    def applies(self, inp: ScanInput) -> bool:
        return bool(inp.user_input or inp.untrusted_context or inp.response)

    def _texts(self, inp: ScanInput) -> list[tuple[str, str]]:
        """(channel, text) pairs this detector scans."""
        return [("user", inp.user_input)]

    def _score(self, inp: ScanInput) -> dict[str, tuple[list[float], list[str]]]:
        hits: dict[str, tuple[list[float], list[str]]] = {t: ([], []) for t in self.types}
        for channel, text in self._texts(inp):
            if not text:
                continue
            for rule in self._rules_for(channel):
                if rule.type in hits and rule.pattern.search(text):
                    hits[rule.type][0].append(rule.weight)
                    hits[rule.type][1].append(f"{channel}: {rule.reason}")
        return hits

    def _rules_for(self, channel: str):
        return self._rules

    def detect(self, inp: ScanInput) -> list[Detection]:
        out = []
        for t, (ws, reasons) in self._score(inp).items():
            out.append(Detection(self.module, t, noisy_or(ws), adapter_id(self),
                                 DEFAULT_SEVERITY[t], self.threshold, "; ".join(reasons)))
        return out


class InjectionDetector(_TextDetector):
    def __init__(self, threshold: float = 0.5):
        super().__init__("guard-injection", "guard",
                         ("direct_injection", "indirect_injection", "context_manipulation",
                          "hierarchy_violation", "obfuscated_instruction"), threshold)

    def _texts(self, inp):
        out = [("user", inp.user_input)] + [("untrusted", c) for c in inp.untrusted_context]
        for h in inp.history:
            if h.get("role") in ("tool_result", "tool", "observation", "retrieved"):
                out.append(("untrusted", str(h.get("content", ""))))
        return out

    def _rules_for(self, channel):
        return INDIRECT_RULES + INJECTION_RULES[2:4] if channel == "untrusted" else INJECTION_RULES

    def _score(self, inp):
        hits = super()._score(inp)
        for channel, text in self._texts(inp):
            if not text:
                continue
            if _ZERO_WIDTH.search(text):
                hits["obfuscated_instruction"][0].append(0.4)
                hits["obfuscated_instruction"][1].append(f"{channel}: zero-width characters")
            for dec in _decoded_base64(text):
                if any(r.pattern.search(dec) for r in INJECTION_RULES + INDIRECT_RULES):
                    hits["obfuscated_instruction"][0].append(0.85)
                    hits["obfuscated_instruction"][1].append(
                        f"{channel}: base64-encoded instruction")
        return hits


class JailbreakDetector(_TextDetector):
    def __init__(self, threshold: float = 0.5):
        super().__init__("guard-jailbreak", "guard", ("jailbreak",), threshold, _rules=JAILBREAK_RULES)


class MisuseDetector(_TextDetector):
    def __init__(self, threshold: float = 0.5):
        super().__init__("misuse-request", "misuse",
                         ("cyber_abuse", "phishing_fraud", "deceptive_content", "general_harm"),
                         threshold, _rules=MISUSE_RULES)

    def applies(self, inp):
        return bool(inp.user_input)


def misuse_hits(text: str) -> list[Rule]:
    """Misuse rules that match ``text`` (used by agent-precheck for ``malicious_request``)."""
    return [r for r in MISUSE_RULES if r.pattern.search(text)]


def injection_hits(text: str) -> list[Rule]:
    return [r for r in INDIRECT_RULES + INJECTION_RULES if r.pattern.search(text)]
