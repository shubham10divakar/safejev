# SafeJEV — implementation plan

Spec: [`../safejev-design-doc.md`](../safejev-design-doc.md). Aims: [`00_aims.md`](00_aims.md).
Running log of decisions and deviations: [`notes.md`](notes.md).

## Layout (`code_repo/safejev/`, own git repo, pushed to github.com/shubham10divakar/safejev)

```
safejev/
  safejev-design-doc.md
  plan/                 00_aims.md  01_implementation_plan.md  notes.md
  safejev/
    __init__.py         SafeJEV, ScanResult, Detection, __version__
    engine.py           SafeJEV: scan(), guard_tool(), from_pretrained()
    core/
      taxonomy.py       decisions, severities, types per module (§7)
      schema.py         Record / Label / Source dataclasses, validation, JSONL io (§6)
      result.py         Detection, ScanResult
      fusion.py         max-severity fusion (§5.4)
      calibration.py    temperature scaling, ECE (pure numpy-free + optional numpy)
      registry.py       adapter registry: user-facing vs research adapters (§5.2)
      detector.py       Detector protocol + ScanInput
    detectors/
      heuristic.py      rule baselines: guard-injection, guard-jailbreak, misuse-request,
                        agent-precheck (needs no weights)
      encoder.py        cross-encoder + LoRA adapters; heads: decision / types / severity
      text.py           render a ScanInput / Record into the model's text pair
    guard/ misuse/ agent/   thin module namespaces exposing default adapters
    governance/
      policy.py         Policy (YAML / Python): (type, severity, tool) → decision
      authz.py          allowlists, argument constraints, rate limits
      approval.py       CLI / callback / webhook approvers, timeout → deny
      audit.py          append-only JSONL audit log, input hash
      report.py         summaries over an audit log
    adapters_io/        openai.py langchain.py mcp.py trajectory.py
    middleware/         decorator.py  mcp_proxy.py
    evaluation/         metrics.py  bootstrap.py  harness.py  latency.py
    cli.py
  training/             (not shipped in the wheel)
    sources.py          source registry, train-ok / eval-only tags, licences (§9.1)
    converters.py       TS-Bench, R-Judge, HINTBench, generic → Record; trajectory → steps (§9.3)
    mutations.py        mutation generator + hard negatives (§9.4)
    leakage.py          exact + MinHash near-dup, canary check, group splits (§9.5)
    train_adapter.py    train one adapter (or joint) from Record JSONL (§10)
    calibrate.py        fit temperature on the calibration split (§10, E6)
  configs/              default_policy.yaml, adapters/*.yaml
  examples/             basic_scan.py guard_tools.py
  tests/
```

## Steps (commit after each; push at milestones)

| # | Step | Design § | Status |
|---|---|---|---|
| 0 | git init, exclude from outer repo, aims + plan + notes | §1 | done |
| 1 | Scaffold: pyproject (light core, `[model]` / `[train]` extras), LICENSE, .gitignore | §13 | done |
| 2 | Core: taxonomy, schema, result, fusion, calibration, registry, detector protocol | §5–7 | done |
| 3 | Heuristic detectors for all four user-facing adapters (+ text rendering) | §5.2 | done |
| 4 | Governance: policy, authz, approval, audit, report | §12 | done |
| 5 | Engine: `SafeJEV.scan`, `guard_tool` with fail-closed, `from_pretrained` | §4, §8 | done |
| 6 | Input converters (OpenAI / LangChain / MCP / trajectory) + MCP stdio proxy | §4 | done |
| 7 | Encoder detector: backbone + LoRA + 3 heads, save / load, temperature | §5.3, §10 | done |
| 8 | Training tooling: sources, converters, mutation generator, leakage, trainer, calibrate | §9, §10 | done |
| 9 | Evaluation: metrics, bootstrap CI, harness (E1/E2/E6), latency (E7) | §11 | |
| 10 | CLI: scan / proxy / eval / report | §8 | |
| 11 | Examples, README, final notes; push | — | |

## Resume here (next session)

(filled in at the end of the round)
