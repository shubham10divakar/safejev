# SafeJEV — Design Document

**Status:** Draft v0.1 · **Date:** 2026-11-06 · **Owner:** Subham
**Sibling project:** CyberJEV (application-layer security checks, built on nano-jev)

---

## 1. Summary

SafeJEV is a pip-installable, modular toolkit for **AI security and governance**. It detects and helps mitigate risks in LLM applications and autonomous agents. It uses small, locally runnable, calibrated classifiers, and a deterministic policy layer decides what happens with each detection.

```
pip install safejev
```

SafeJEV does **not** claim to make an LLM safe. It provides measurable detection, configurable enforcement and transparent audit.

---

## 2. Scope

The four standard LLM threat classes, and which ones SafeJEV covers:

| # | Threat class | In scope? | SafeJEV module |
|---|---|---|---|
| 1 | Inference-time attacks via prompt manipulation (injection, jailbreak, context hijack) | **Yes** | `safejev.guard` |
| 2 | Training-time attacks (data poisoning, backdoors) | **No (future work)** | — |
| 3 | Misuse by malicious actors (phishing, malicious code, disinformation) | **Yes** | `safejev.misuse` |
| 4 | Intrinsic risks of LLM-based autonomous agents | **Yes** | `safejev.agent` |

Why #2 is excluded: a training-time attack is planted before deployment. Detecting it needs access to the training data or model internals, not the inference-time inputs SafeJEV sees. A possible later module is `safejev.audit_data` (dataset hygiene and trigger scanning). It is not part of v0.x.

### Goals
- Local inference by default, so prompts and trajectories never leave the user's machine.
- Fast enough to run **before every tool call** (target: p95 < 50 ms on CPU for encoder adapters).
- Calibrated scores, so thresholds mean something (continuing the CyberJEV approach).
- Every component can be installed, tested and versioned on its own.

### Non-goals
- Fact-checking. `misuse.disinformation` detects *intent to generate deceptive content*, not whether a claim is true.
- Replacing authorization. High-impact actions still need deterministic permission checks, independent of any model.
- Being a general content moderator. SafeJEV focuses on security, not toxicity or style.

---

## 3. Relationship to CyberJEV

Both projects share the same idea: a small typed-decision cross-encoder whose output feeds a policy layer. I suggest pulling the shared parts into a small common package rather than copying code:

```
jev-core     → backbone loading, adapter management, calibration, decision schema, audit format
cyberjev     → depends on jev-core; app-layer security checks
safejev      → depends on jev-core; AI security & governance
```

A user who installs both gets one backbone in memory and adds adapters for each package.

---

## 4. Architecture

```
            ┌───────────────────────────────────────────────────────┐
 inputs ──▶ │ Input adapters                                        │
            │  prompt · response · retrieved doc · tool call ·       │
            │  trajectory (OpenAI / LangChain / MCP formats)         │
            └──────────────┬────────────────────────────────────────┘
                           ▼
            ┌───────────────────────────────────────────────────────┐
            │ Detectors (shared backbone + task adapters)           │
            │  guard.*        misuse.*         agent.*              │
            └──────────────┬────────────────────────────────────────┘
                           ▼  calibrated scores + types + provenance
            ┌───────────────────────────────────────────────────────┐
            │ Fusion — combine detections → risk level              │
            └──────────────┬────────────────────────────────────────┘
                           ▼
            ┌───────────────────────────────────────────────────────┐
            │ Governance (deterministic, no ML)                     │
            │  policies · authorization rules · human approval ·    │
            │  audit log · reports                                   │
            └──────────────┬────────────────────────────────────────┘
                           ▼
              ALLOW · ASK (human approval) · BLOCK · REPAIR-HINT
```

### Two deployment modes
1. **Standalone:** scan prompts, responses or full trajectories, and produce reports.
2. **Middleware:** intercept tool calls before they execute. It ships as:
   - a Python decorator / wrapper for tool functions,
   - adapters for common agent frameworks,
   - an **MCP proxy** that sits between an MCP client and its servers and screens every `tools/call`. This is the highest-leverage integration, because one proxy covers any MCP-based agent.

**Fail-closed rule:** in middleware mode, if the detector is unavailable or errors, any tool marked `sensitive` is blocked or sent for approval, never silently allowed.

---

## 5. Model strategy: separate weights

### 5.1 The decision

User-selectable separate weights are a good idea. They should be built as **one shared backbone plus small LoRA adapters**, not as many independently fine-tuned full models.

| Option | Pros | Cons |
|---|---|---|
| A. One model trained jointly on everything | Simple; shared learning | One task can degrade another; hard to update one domain; no user choice |
| B. A separate full model per dataset | Maximal isolation | N × backbone size on disk/RAM; each model learns one dataset's annotation habits; small datasets (R-Judge ≈ 569, HINTBench ≈ 629) make weak standalone models |
| **C. Shared backbone + adapters (chosen)** | Few MB per adapter; user picks adapters; one backbone in memory; joint vs. separate is just a training config, so the research comparison is cheap | Slightly more engineering (adapter loading, fusion) |

### 5.2 What users choose: capability adapters, not dataset adapters

A developer knows their threat ("my agent calls tools"), not which benchmark matches their deployment. So the **user-facing** choice is by capability. The per-dataset adapters are still trained and published, but marked as research artifacts.

**User-facing adapters (deployment):**

| Adapter | Module | Detects |
|---|---|---|
| `guard-injection` | guard | Direct + indirect prompt injection, instruction-hierarchy violation |
| `guard-jailbreak` | guard | Attempts to bypass safety constraints |
| `misuse-request` | misuse | Harmful requests incl. cyber abuse, phishing/fraud, deceptive content |
| `agent-precheck` | agent | Pre-execution tool-call check: risky args, harmful tool, injected-goal call, intent mismatch, malformed call |
| `agent-trajectory` | agent | Multi-step risk across a trajectory |

**Research adapters (published, labeled "research"):** `agent-tsbench`, `agent-hintbench`, `agent-auragen`, `agent-synthetic-mutations`, etc. Each is trained on one source, and they are used for the ablations in §11.

Users can still load any adapter explicitly (§8).

### 5.3 Backbone choice
- **v0.1:** an encoder cross-encoder (nano-jev, or DeBERTa-v3 / ModernBERT class). It's fast, easy to calibrate, and runs on CPU.
- **Long-context variant for `agent-trajectory`:** a small decoder with a classification head plus LoRA (≈0.5–2B class). Agent trajectories can be long; HINTBench averages ~33 steps.
- **Optional explainer:** a small generative model that writes a rationale *only when asked*. It is never on the hot path.

### 5.4 Fusion
v0.1 has no learned router. The user's config decides which adapters run. Fusion takes each adapter's calibrated score against its own threshold and reports the **highest-severity** detection plus all detections. A learned router is an experiment (E5), not a v0.1 feature.

---

## 6. Unified data schema

Every sample from every source is normalized into one record type. Provenance and original labels are always kept.

```json
{
  "id": "tsbench-train-000123-step4",
  "source": {"dataset": "TS-Bench", "split": "train", "version": "v1", "license": "…"},
  "granularity": "step",
  "module": "agent",
  "input": {
    "user_goal": "Find duplicate files in /documents",
    "history": [{"role": "tool_call", "content": "..."}, {"role": "tool_result", "content": "..."}],
    "untrusted_context": ["...retrieved webpage text..."],
    "proposed_action": {"tool": "delete_file", "arguments": {"path": "/documents/report.pdf"}},
    "tool_schema": {"name": "delete_file", "params": {"path": "string"}, "sensitive": true}
  },
  "label": {
    "decision": "ask",
    "types": ["intent_mismatch", "destructive_action"],
    "severity": "high"
  },
  "original_label": {"raw": "controversial", "scheme": "TS-Bench safe/controversial/unsafe"},
  "meta": {"risky_step_index": 4, "generator": null, "human_verified": false}
}
```

---

## 7. Label taxonomy

**Decision (all modules):** `allow` · `ask` · `block`

**Types (multi-label):**

| Module | Types |
|---|---|
| guard | `direct_injection`, `indirect_injection`, `jailbreak`, `context_manipulation`, `hierarchy_violation`, `obfuscated_instruction` |
| misuse | `cyber_abuse`, `phishing_fraud`, `deceptive_content`, `general_harm` |
| agent | `malicious_request`, `injected_goal`, `harmful_tool`, `risky_arguments`, `intent_mismatch`, `unauthorized_action`, `data_exfiltration`, `destructive_action`, `malformed_call`, `multi_step_risk` |

The agent module keeps the distinction between **externally induced** risk (malicious user, injection) and **intrinsic** risk (the agent's own mistakes on benign tasks, as in HINTBench). It is reported as a separate metric slice.

**Mapping rules:** a TS-Bench "controversial" label maps to `ask`. R-Judge's binary trajectory label maps to `granularity: trajectory`, and its step label is derived as in §9.3.

---

## 8. Python API

### Basic scan
```python
from safejev import SafeJEV

sj = SafeJEV(modules=["guard", "agent"])          # loads default capability adapters

result = sj.scan(
    user_input="Find duplicate files in /documents",
    agent_action={"tool": "delete_file",
                  "arguments": {"path": "/documents/report.pdf"}},
)
print(result.decision, result.risk_level)
for d in result.detections:
    print(d.module, d.type, d.score, d.adapter)
```

Example output (illustrative only, not measured):
```json
{
  "decision": "ask",
  "risk_level": "high",
  "detections": [
    {"module": "agent", "type": "intent_mismatch", "score": 0.91,
     "adapter": "safejev/agent-precheck@0.1.0",
     "reason": "User asked to find duplicates; action deletes a file."}
  ],
  "latency_ms": 18
}
```

### Explicit adapter selection
```python
sj = SafeJEV.from_pretrained(
    backbone="safejev/backbone-base",
    adapters=["safejev/guard-injection", "safejev/agent-tsbench"],   # research adapter, explicitly chosen
)
```

### Middleware: guarding tools
```python
from safejev import SafeJEV
from safejev.governance import Policy

policy = Policy(
    block_on=["critical"],
    ask_on=["high"],
    require_approval_tools=["delete_file", "send_email", "transfer_funds", "external_upload"],
    fail_closed=True,
    audit_log="./safejev_audit.jsonl",
)
sj = SafeJEV(modules=["agent", "guard"], policy=policy)

@sj.guard_tool(sensitive=True)
def delete_file(path: str): ...
```

### CLI
```
safejev scan --input prompt.txt --modules guard
safejev proxy --mcp-config servers.json          # MCP screening proxy
safejev eval --adapter safejev/agent-precheck --benchmark pre-exec-bench
safejev report --audit safejev_audit.jsonl
```

Weights download lazily from the Hugging Face Hub. `pip install safejev` pulls only code.

---

## 9. Data plan

### 9.1 Rule: train/eval separation
Several public benchmarks ask not to be used for training, or are needed as clean test sets. **Every source is tagged `train-ok` or `eval-only` before any training.** Licenses must be confirmed per source. The table below is a plan; entries marked † are from memory and have not been checked yet.

### 9.2 Sources per module

**guard**

| Source | Use | Notes |
|---|---|---|
| Public prompt-injection sets (e.g. deepset/prompt-injections†, SPML†, BIPIA† for indirect) | train-ok† | Small; check license |
| In-the-wild jailbreak prompt collections† | train-ok† | Dedupe heavily |
| Own synthetic injections (templates × carriers × obfuscations: encoding, multilingual, split-turn) | train | Main volume |
| JailbreakBench† | **eval-only** | |
| AgentDojo, InjecAgent | **eval-only** | Agent-loop evaluation (ASR + utility) |

**misuse**

| Source | Use | Notes |
|---|---|---|
| Open guard-training mixes (e.g. WildGuardMix†, Aegis 2.0†) | train-ok† | Check licenses; some are non-commercial |
| Phishing/fraud email corpora† | train-ok† | For `phishing_fraud` |
| Synthetic deceptive-content requests | train | Little public data exists; human-verify a test slice |
| HarmBench†, StrongREJECT†, CyberSecEval† | **eval-only** | |
| AgentHarm | **eval-only** | Dataset card asks that it not be used for training |

**agent**

| Source | Use | Notes |
|---|---|---|
| TS-Bench **train split** | train | Step-level; includes a "controversial" label |
| AuraGen recipe (Safiron) | train | Synthesize benign trajectories → inject typed risks → reward-filter |
| Own mutation generator over correct function calls (BFCL†, xLAM†, ToolBench†, τ-bench†) | train | Main source for `intent_mismatch`, `malformed_call`, `risky_arguments` |
| Hard negatives (scary-looking but authorized calls) | train | Main defence against false positives |
| TS-Bench **eval split** (AgentHarm-Traj, ASB-Traj, AgentDojo-Traj) | **eval-only** | |
| Pre-Exec Bench | **eval-only** | Compare against Safiron |
| R-Judge | eval (default) | Small; also used in leave-one-out (E2) |
| HINTBench | eval (default) | Intrinsic-risk slice; also used in leave-one-out (E2) |

### 9.3 Converting trajectories to step samples
Cut each trajectory at every tool call, giving `(goal, history so far, proposed call) → label`. Where only trajectory-level labels exist (R-Judge), identify the risky step with an LLM pass and then verify it by hand. Steps before it are labeled `allow`; the risky step gets the trajectory's label.

### 9.4 Mutation generator (agent)
Start from correct calls and apply typed edits. The label comes from the edit:
- argument swap (recipient, account, path) → `intent_mismatch`
- scope widening (`id=42` → `*`; read → delete) → `risky_arguments` / `destructive_action`
- value corruption (amount ×10, currency, units, date) → `intent_mismatch`
- wrong tool / invented parameter → `malformed_call`
- planted instruction in a tool result, followed by the obeying call → `injected_goal`
- sending data to an unlisted external endpoint → `data_exfiltration`
- **hard negatives:** a destructive or external call that the user explicitly asked for → `allow`

Use an LLM only to paraphrase user goals for diversity, and filter its output with a verifier. Human-verify about 500 test samples per module.

### 9.5 Leakage controls
- Deduplicate train against all eval sets, both exact and near-duplicate (MinHash on normalized text).
- Check for benchmark canary strings and refuse to train on any file that contains one.
- Split by **tool / domain / scenario**, not at random. Hold out one planner model entirely.

---

## 10. Training plan

| Item | Plan |
|---|---|
| Hardware | Kaggle T4/P100 (encoder + LoRA fits comfortably) |
| Stage 1 | Train each adapter separately (one per capability, plus one per source for research) |
| Stage 2 | Joint multi-task training, for comparison |
| Heads | Decision head (3-way) + multi-label type head + severity |
| Loss | CE (decision) + BCE (types); class-balanced weights |
| Calibration | Per-adapter temperature scaling on a held-out calibration split; report ECE before and after |
| Versioning | Each adapter is versioned semantically and ships a model card: training sources, licenses, eval results, known failure modes |

---

## 11. Experiments (designed before training)

### E1 — Separate vs. joint vs. adapters
**RQ:** Does shared learning help, or does specialization win?
Configs: base model (zero-shot) · per-source adapters · per-capability adapters · joint multi-task full fine-tune · joint LoRA · ensemble of per-source adapters.
All configs are evaluated on the same held-out test sets.

### E2 — Cross-benchmark generalization (leave-one-benchmark-out)
**RQ:** Does the detector transfer to agent frameworks and annotation styles it never saw?

| Train on | Test on |
|---|---|
| everything except Pre-Exec Bench | Pre-Exec Bench |
| everything except R-Judge | R-Judge |
| everything except HINTBench | HINTBench |
| everything except AgentDojo-derived data | AgentDojo-Traj |
| synthetic only | all real benchmarks |

The last row measures how much the generator alone is worth.

### E3 — Adversarial robustness (guard)
**RQ:** Does detection hold against attacks absent from training, and against an attacker who adapts to the detector?
Held-out attack families: obfuscation/encoding, multilingual, multi-turn, unseen templates. Adaptive attack: the attacker optimizes against the detector score (black-box query attack plus LLM paraphrase search).
Metrics: detection rate, FPR on benign inputs, ASR in AgentDojo/InjecAgent loops, **benign task completion**. A low ASR only counts if utility holds.

### E4 — Step vs. context vs. trajectory (agent)
**RQ:** How much does history add?
Inputs: proposed call only · call + goal · call + goal + last k steps (k = 1, 3, all) · full trajectory (decoder variant).
Also measures *how early* risk is caught: the step index of the first `ask`/`block` compared with the annotated risky step.

### E5 — Fusion and routing
**RQ:** Is a learned router better than running all selected adapters and taking the highest-severity result?
Compare: run all + max-severity · learned router · cascade (cheap adapter first, escalate to trajectory model on uncertainty).

### E6 — Calibration
**RQ:** Are scores trustworthy as thresholds?
ECE and reliability diagrams per adapter, before and after temperature scaling, in-distribution vs. on E2 held-out benchmarks.

### E7 — Cost
Latency (p50/p95, CPU and GPU), throughput, memory, as a function of number of adapters loaded, plus MCP-proxy overhead per tool call.

### Baselines
TS-Guard (open, Apache-2.0), Safiron, LlamaGuard-class models, PromptGuard/LlamaFirewall-class detectors†, and a zero-shot LLM-as-judge.

### Common metrics
Precision, recall, F1, **recall @ fixed FPR (1%, 5%)**, AUROC, AUPRC, ECE, ASR, utility retention, latency.
Agent-specific: unauthorized-action rate, exfiltration rate, destructive-action rate, legitimate task completion, approval frequency (how often a human gets asked). All results are also reported split into intrinsic vs. externally induced risk.

---

## 12. Governance layer

Deterministic and auditable. No ML here.

- **Policies:** YAML/Python rules mapping (type, severity, tool) to a decision.
- **Authorization:** per-tool allowlists, argument constraints (paths, domains, amount limits), and rate limits. These run even when the model says `allow`.
- **Human approval:** pluggable approver (CLI prompt, webhook, Slack), with timeout → deny.
- **Audit log:** append-only JSONL. Each entry records input hash, adapters + versions, scores, decision, policy rule hit and approver.
- **Reports:** per-period summaries (detections by type, approval load, FPR estimates from overridden blocks).
- Scores are treated as calibrated only for adapters whose model card reports passing E6.

---

## 13. Package layout

```
safejev/
├── safejev/
│   ├── __init__.py
│   ├── core/          # engine, schema, adapter registry/loader, fusion, calibration
│   ├── adapters_io/   # input converters: OpenAI, LangChain, MCP, raw trajectories
│   ├── guard/
│   ├── misuse/
│   ├── agent/
│   ├── governance/    # policy, authorization, approvals, audit, reports
│   ├── middleware/    # decorators, framework hooks, MCP proxy
│   ├── evaluation/    # metrics, benchmark loaders, adversarial suite, report gen
│   └── cli.py
├── training/          # data normalization, mutation generator, train configs (not shipped in wheel)
├── tests/
├── examples/
├── configs/
├── pyproject.toml
└── README.md
```

Releases:
- **PyPI:** `safejev` (code only)
- **Hugging Face:** `safejev/backbone-*` plus one repo per adapter, each with a model card
- **GitHub:** training configs, normalized-schema converters, evaluation harness, results tables

---

## 14. Roadmap

| Version | Content | Paper hook |
|---|---|---|
| **v0.1** | jev-core, schema, `agent-precheck` (encoder), mutation generator, governance + decorator, eval harness for E1/E2/E6/E7 | Main paper: calibrated pre-execution check incl. intent mismatch, cross-benchmark generalization |
| v0.2 | `guard-injection`, `guard-jailbreak`, MCP proxy, E3 | Robustness + security/utility in agent loops |
| v0.3 | `agent-trajectory` (decoder), E4, E5 | Step vs. trajectory context; router vs. ensemble |
| v0.4 | `misuse-request`, reports | Toolkit / systems paper |
| later | `audit_data` (training-time attacks) | Future work |

**Why the agent module comes first:** prompt-injection and jailbreak detectors already exist, while a pre-execution check covering intent mismatch is the open gap. `guard-injection` follows immediately, because injected goals are an agent risk too.

---

## 15. Risks and open questions

- **Licenses:** confirm every source before training, and record each one in the schema's `source.license`.
- **Synthetic-data bias:** a model trained mostly on generated mutations may learn the generator's quirks. The synthetic-only row of E2 measures this directly.
- **Small benchmarks:** R-Judge and HINTBench are small and imbalanced. Report confidence intervals (bootstrap).
- **Over-blocking:** the approval load must stay usable. Track approval frequency as a first-class metric.
- **Open:** keep jev-core as a separate package, or vendor it into both projects?
- **Open:** default backbone (nano-jev vs. a public encoder)? Decide after a small pilot on TS-Bench-train.

---

## 16. References (verify before citing)

- ToolSafe / TS-Bench / TS-Guard — arXiv 2601.10156
- Safiron / AuraGen / Pre-Exec Bench — arXiv 2510.09781
- R-Judge — arXiv 2401.10019
- HINTBench — arXiv 2604.13954
- AgentHarm — arXiv 2410.09024 (eval-only per dataset card)
- AgentDojo, InjecAgent — prompt-injection agent benchmarks
- Arp et al., "Dos and Don'ts of Machine Learning in Computer Security"
