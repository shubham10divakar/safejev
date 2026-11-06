# SafeJEV — aims

Spec: [`../safejev-design-doc.md`](../safejev-design-doc.md) (§ numbers below refer to it).

## What v0.1 must deliver (§14)

1. **Unified schema + taxonomy** (§6, §7): one record type for every source, with provenance
   and the original label kept.
2. **Engine** (§4, §5.4, §8): `SafeJEV(...).scan(...)` runs the selected detectors, fuses them
   (highest severity wins, all detections reported) and returns `allow / ask / block`.
3. **Detectors behind one interface**: a deterministic heuristic baseline that works with no
   weights, and an encoder cross-encoder with LoRA adapters (decision + type + severity heads),
   loaded lazily from disk or the Hugging Face Hub.
4. **Governance** (§12): deterministic policy, authorization rules, human approval with
   timeout → deny, append-only JSONL audit log, reports. No ML here.
5. **Middleware** (§4): `@sj.guard_tool(...)` decorator with the fail-closed rule, plus input
   converters (OpenAI, LangChain, MCP) and an MCP stdio screening proxy.
6. **Training tooling** (§9, §10): source registry (train-ok / eval-only), converters, the
   mutation generator (§9.4), leakage controls (§9.5), adapter trainer, temperature scaling.
7. **Evaluation harness** (§11 E1/E2/E6/E7): the common metrics, bootstrap CIs, leave-one-out
   splits, latency bench.
8. **CLI** (§8): `scan`, `proxy`, `eval`, `report`.

## Scope of this round (2026-11-06)

Implement and test on CPU only. No dataset downloads, no training runs, no published weights.
Tests use a tiny randomly initialised BERT config, so they need no network.

## Success criteria for the code round

- `pip install -e .` gives a working `safejev` with only light dependencies (`pyyaml`).
  Torch / transformers / peft are an optional extra (`safejev[model]`).
- Every public API in §8 runs end to end with the heuristic detectors.
- The encoder adapter trains for a few steps on synthetic data, saves, reloads and gives the
  same scores (round-trip test).
- The test suite passes on CPU without network.

## Kill / rethink criteria (for the training round, later)

- `agent-precheck` on TS-Bench eval does not beat the heuristic baseline by a clear margin
  (AUROC + 0.05) → rethink the data mix before anything else.
- Synthetic-only training (E2 last row) is within noise of the full mix → the real data is not
  adding value, or the generator dominates; investigate.
- Approval frequency on benign trajectories > 20% at the default policy → over-blocking (§15).
