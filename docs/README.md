# SemanticFlowSR Docs

Current docs are maintained for:

```text
Edge-Parameterized Semantic Flow Matching
```

Start here:

| Document | Purpose |
|---|---|
| [../AGENTS.md](../AGENTS.md) | New-session handoff: environment, algorithm, commands, current limits. |
| [../README.md](../README.md) | Human quick start and command summary. |
| [ALGORITHM.md](ALGORITHM.md) | Current Edge Flow theory and implementation contract. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Code layout for `semflow_sr/edge_flow/`. |
| [RESULTS.md](RESULTS.md) | Current smoke-only result entry. |
| [IMPROVEMENT_NOTES.md](IMPROVEMENT_NOTES.md) | Current bottlenecks and next Edge Flow milestones. |

Dataset and baseline docs remain useful for future benchmark integration:

| Document | Purpose |
|---|---|
| [datasets/overview.md](datasets/overview.md) | Dataset fields and local layout. |
| [datasets/adaptation.md](datasets/adaptation.md) | Unified benchmark manifest and suite adaptation. |
| [datasets/deployment.md](datasets/deployment.md) | Dataset build/deploy commands. |
| [baselines/README.md](baselines/README.md) | External baseline matrix and result fields. |

Current command entry points:

```text
semflow_sr/edge_flow/train_edge_flow.py
scripts/run_edge_flow.py
scripts/validate_benchmark_manifest.py
```

Legacy diagnostic entry points:

```text
semflow_sr/train/train_path_posterior_flow.py
scripts/run_path_posterior_flow.py
semflow_sr/train/train_block_flow.py
scripts/run_block_risk_flow.py
```

`results/edge_flow_smoke/` is a wiring check only, not a benchmark result.
