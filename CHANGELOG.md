# Changelog

## 0.1.0 — 2026-11-06

First release (research preview). Code only; no trained weights are published yet.

- Unified record schema and label taxonomy for guard / misuse / agent.
- Heuristic detectors (no weights needed): `guard-injection`, `guard-jailbreak`,
  `misuse-request`, `agent-precheck` (intent mismatch, risky arguments, destructive actions,
  injected goals, exfiltration, malformed calls, multi-step risk), with hard-negative handling.
- Encoder detector: shared backbone + LoRA adapters, decision / type / severity heads,
  per-head temperature scaling; several adapters share one backbone in memory.
- Fusion (highest severity wins) and a deterministic policy layer: rules, severity thresholds,
  authorization (allowlists, argument constraints, rate limits), fail-closed for sensitive tools.
- Human approval (CLI, callback, webhook; timeout means deny), append-only JSONL audit log,
  audit reports.
- `SafeJEV.guard_tool` decorator (sync and async) with sessions; OpenAI, LangChain, MCP and
  trajectory input converters; MCP stdio screening proxy and config wrapper.
- Evaluation: precision / recall / F1, recall at fixed FPR, AUROC (bootstrap CI), AUPRC, ECE,
  approval frequency, intrinsic vs. external slices, latency.
- CLI: `safejev scan | proxy | eval | report | adapters`.
- Training tools in the source distribution (`training/`): source gate (train-ok / eval-only),
  converters, mutation generator, leakage controls, LoRA trainer, calibration.
