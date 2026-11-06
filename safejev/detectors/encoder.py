"""Encoder cross-encoder detector: one shared backbone + a LoRA adapter and three heads per
capability (design doc §5.1, §5.3, §10).

Heads on the pooled [CLS] vector:
- decision: 3-way softmax (allow / ask / block)
- types:    multi-label sigmoid over the module's types
- severity: 5-way softmax (none ... critical)

Each head has its own temperature (fitted on a calibration split, §10 / E6).

On-disk layout of an adapter directory::

    safejev_adapter.json     metadata: name, module, version, backbone, types, temperatures, ...
    adapter_config.json      \
    adapter_model.safetensors > the peft LoRA adapter (absent when lora=false: full fine-tune)
    heads.pt                 state dict of the three heads
    backbone/                (only for full fine-tunes) the whole fine-tuned backbone

Many adapters on the same backbone share one copy of it in memory (``_BackboneCache``): each
adapter is loaded into the same ``PeftModel`` under its own name and activated per call.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError("the encoder backend needs torch, transformers and peft: "
                      "pip install 'safejev[model]'") from e

from ..core.calibration import sigmoid, softmax
from ..core.detector import ScanInput, adapter_id
from ..core.result import Detection
from ..core.taxonomy import DECISIONS, DEFAULT_SEVERITY, MODULE_TYPES, SEVERITIES
from .text import render_pair

META_FILE = "safejev_adapter.json"
HEADS_FILE = "heads.pt"

# LoRA target modules by backbone family; falls back to "all-linear".
LORA_TARGETS = {
    "bert": ["query", "value"], "roberta": ["query", "value"], "xlm-roberta": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"], "modernbert": ["Wqkv"],
    "distilbert": ["q_lin", "v_lin"], "electra": ["query", "value"],
}


@dataclass
class AdapterMeta:
    name: str
    module: str
    backbone: str
    version: str = "0.1.0"
    types: list[str] = field(default_factory=list)
    granularity: str = "step"
    max_length: int = 512
    last_k: int | None = None              # history steps rendered (E4); None = all
    lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] | str | None = None
    t_decision: float = 1.0
    t_types: list[float] = field(default_factory=list)
    t_severity: float = 1.0
    threshold: float = 0.5
    type_thresholds: dict[str, float] = field(default_factory=dict)
    use_severity_head: bool = False
    tier: str = "user"
    training_sources: list[str] = field(default_factory=list)
    calibrated: bool = False               # True once temperatures were fitted (E6)

    def __post_init__(self):
        if not self.types:
            self.types = list(MODULE_TYPES[self.module])
        if not self.t_types:
            self.t_types = [1.0] * len(self.types)

    def save(self, d: Path):
        (d / META_FILE).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "AdapterMeta":
        return cls(**json.loads((d / META_FILE).read_text(encoding="utf-8")))


class Heads(nn.Module):
    def __init__(self, hidden: int, n_types: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.decision = nn.Linear(hidden, len(DECISIONS))
        self.types = nn.Linear(hidden, n_types)
        self.severity = nn.Linear(hidden, len(SEVERITIES))

    def forward(self, pooled):
        h = self.drop(pooled)
        return self.decision(h), self.types(h), self.severity(h)


def pool(out, attention_mask):
    """The [CLS] (first-token) vector."""
    return out.last_hidden_state[:, 0]


def lora_targets(model, override=None):
    if override:
        return override
    mt = getattr(getattr(model, "config", None), "model_type", "")
    return LORA_TARGETS.get(mt, "all-linear")


def default_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class _BackboneCache:
    """One backbone (and tokenizer) per (id, device); adapters attach to it by name."""

    def __init__(self):
        self._items: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, backbone: str, device: str):
        from transformers import AutoModel, AutoTokenizer
        key = (backbone, device)
        with self._lock:
            if key not in self._items:
                tok = AutoTokenizer.from_pretrained(backbone)
                model = AutoModel.from_pretrained(backbone).to(device).eval()
                self._items[key] = {"tok": tok, "base": model, "peft": None,
                                    "lock": threading.Lock()}
            return self._items[key]

    def attach(self, item, adapter_dir: Path, name: str):
        from peft import PeftModel
        with item["lock"]:
            if item["peft"] is None:
                item["peft"] = PeftModel.from_pretrained(item["base"], str(adapter_dir),
                                                         adapter_name=name).eval()
            elif name not in item["peft"].peft_config:
                item["peft"].load_adapter(str(adapter_dir), adapter_name=name)
            item["peft"].eval()

    def clear(self):
        with self._lock:
            self._items.clear()


BACKBONES = _BackboneCache()


def _resolve_backbone(meta_backbone: str, adapter_dir: Path) -> str:
    override = os.environ.get("SAFEJEV_BACKBONE_PATH")
    if override:
        return override
    local = adapter_dir / "backbone"
    if local.is_dir():
        return str(local)
    rel = adapter_dir / meta_backbone
    return str(rel) if rel.is_dir() else meta_backbone


class EncoderDetector:
    """A trained adapter. ``detect`` returns one Detection per type (calibrated if fitted)."""

    def __init__(self, meta: AdapterMeta, model, heads: Heads, tokenizer, device: str,
                 lock: threading.Lock | None = None, peft_name: str | None = None):
        self.meta = meta
        self.name, self.module = meta.name, meta.module
        self.version = meta.version
        self.threshold = meta.threshold
        self.model, self.heads, self.tok = model, heads.to(device).eval(), tokenizer
        self.device = device
        self._lock = lock or threading.Lock()
        self._peft_name = peft_name

    # ---- loading ---------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, device: str | None = None, share_backbone: bool = True
             ) -> "EncoderDetector":
        d = Path(path)
        meta = AdapterMeta.load(d)
        device = default_device(device)
        state = torch.load(d / HEADS_FILE, map_location="cpu", weights_only=True)
        hidden = state["decision.weight"].shape[1]
        heads = Heads(hidden, len(meta.types))
        heads.load_state_dict(state)
        backbone = _resolve_backbone(meta.backbone, d)
        if not meta.lora:
            from transformers import AutoModel, AutoTokenizer
            src = d / "backbone"
            model = AutoModel.from_pretrained(str(src)).to(device).eval()
            tok = AutoTokenizer.from_pretrained(str(src) if (src / "tokenizer_config.json").exists() else backbone)
            return cls(meta, model, heads, tok, device)
        if share_backbone:
            item = BACKBONES.get(backbone, device)
            key = f"{meta.name}@{meta.version}:{d.resolve()}".encode()
            peft_name = "a_" + hashlib.sha1(key).hexdigest()[:12]
            BACKBONES.attach(item, d, peft_name)
            return cls(meta, item["peft"], heads, item["tok"], device, item["lock"], peft_name)
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer
        base = AutoModel.from_pretrained(backbone).to(device)
        model = PeftModel.from_pretrained(base, str(d)).to(device).eval()
        return cls(meta, model, heads, AutoTokenizer.from_pretrained(backbone), device)

    @classmethod
    def from_hub(cls, ref: str, device: str | None = None, cache_dir: str | None = None):
        """'safejev/agent-precheck@0.1.0' → download that repo revision and load it."""
        from huggingface_hub import snapshot_download

        from ..core.registry import parse_ref
        name, ver = parse_ref(ref)
        org = os.environ.get("SAFEJEV_HUB_ORG", "safejev")
        local = snapshot_download(f"{org}/{name}", revision=f"v{ver}" if ver else None,
                                  cache_dir=cache_dir)
        return cls.load(local, device=device)

    # ---- inference -------------------------------------------------------------------------
    def applies(self, inp: ScanInput) -> bool:
        if self.meta.granularity == "text":
            return bool(inp.user_input or inp.untrusted_context or inp.response)
        return bool(inp.agent_action)

    def encode(self, inputs: list[ScanInput]):
        pairs = [render_pair(i, self.meta.granularity, self.meta.last_k) for i in inputs]
        ctx = [p[0] or " " for p in pairs]
        cand = [p[1] or " " for p in pairs]
        return self.tok(ctx, cand, truncation="longest_first", max_length=self.meta.max_length,
                        padding=True, return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def logits(self, inputs: list[ScanInput]):
        enc = self.encode(inputs)
        with self._lock:
            if self._peft_name is not None:
                self.model.set_adapter(self._peft_name)
            out = self.model(**enc)
        return self.heads(pool(out, enc["attention_mask"]))

    def predict(self, inputs: list[ScanInput], batch_size: int = 32) -> list[dict[str, Any]]:
        """Calibrated probabilities for each input: decision, types, severity."""
        m = self.meta
        rows = []
        for s in range(0, len(inputs), batch_size):
            zd, zt, zs = (x.float().cpu().tolist() for x in self.logits(inputs[s:s + batch_size]))
            for d, t, sv in zip(zd, zt, zs):
                rows.append({
                    "decision": dict(zip(DECISIONS, softmax(d, m.t_decision))),
                    "types": {ty: sigmoid(z / tt) for ty, z, tt in zip(m.types, t, m.t_types)},
                    "severity": dict(zip(SEVERITIES, softmax(sv, m.t_severity))),
                })
        return rows

    def detect(self, inp: ScanInput) -> list[Detection]:
        p = self.predict([inp])[0]
        sev_pred = None
        if self.meta.use_severity_head:
            sev = {k: v for k, v in p["severity"].items() if k != "none"}
            sev_pred = max(sev, key=sev.get)
        dec = ", ".join(f"P({k})={v:.2f}" for k, v in p["decision"].items())
        return [Detection(self.module, t, s, adapter_id(self),
                          sev_pred or DEFAULT_SEVERITY[t],
                          self.meta.type_thresholds.get(t, self.threshold), dec)
                for t, s in p["types"].items()]


# ---- building and saving (used by training/train_adapter.py) --------------------------------
def build(meta: AdapterMeta, device: str | None = None, backbone_path: str | None = None):
    """Fresh (model, heads, tokenizer) for training. LoRA unless ``meta.lora`` is False."""
    from transformers import AutoModel, AutoTokenizer
    device = default_device(device)
    src = backbone_path or meta.backbone
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(src)
    if meta.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=meta.lora_r, lora_alpha=meta.lora_alpha, lora_dropout=meta.lora_dropout,
                         target_modules=lora_targets(model, meta.lora_targets))
        model = get_peft_model(model, cfg)          # adapter name "default"
    heads = Heads(model.config.hidden_size, len(meta.types))
    return model.to(device), heads.to(device), tok


def save(out_dir: str | Path, meta: AdapterMeta, model, heads: Heads, tokenizer=None) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta.save(d)
    torch.save({k: v.detach().cpu() for k, v in heads.state_dict().items()}, d / HEADS_FILE)
    if meta.lora:
        model.save_pretrained(str(d))
    else:
        model.save_pretrained(str(d / "backbone"))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(d / "backbone"))
    return d
