# SemanticFlowSR Docs

Current docs are maintained for:

```text
Semantic-Fisher Flow Matching
```

Start here:

| Document | Purpose |
|---|---|
| [../AGENTS.md](../AGENTS.md) | New-session handoff: environment, algorithm, commands, current limits. |
| [../README.md](../README.md) | Human quick start and command summary. |
| [ALGORITHM.md](ALGORITHM.md) | Current algorithm: TargetSampler q_hat -> semantic-Fisher endpoint flow matching. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Code layout and main data flow for the action-support Semantic-Fisher Flow Matching path. |
| [MATH.md](MATH.md) | Centered residual, semantic effect, pullback metric, log-rate solver, sphere update. |
| [THEORY_MAPPING.md](THEORY_MAPPING.md) | Formula-to-code mapping; may include compatibility notes. |
| [IMPROVEMENT_NOTES.md](IMPROVEMENT_NOTES.md) | 当前 action-flow + STOP 的 87-task 指标、问题分析和改进建议。 |

Dataset and baseline docs:

| Document | Purpose |
|---|---|
| [datasets/overview.md](datasets/overview.md) | Dataset fields and local layout. |
| [datasets/adaptation.md](datasets/adaptation.md) | Unified benchmark manifest and suite adaptation. |
| [datasets/deployment.md](datasets/deployment.md) | Dataset build/deploy commands. |
| [baselines/README.md](baselines/README.md) | External baseline matrix, environment boundaries, result fields. |

Historical notes:

| Document | Status |
|---|---|
| [EVAL_87_ADAPTATION.md](EVAL_87_ADAPTATION.md) | Historical 87-task adaptation note; not the current result entry. |

Current command entry points:

```text
semflow_sr/train/train_path_posterior_flow.py
scripts/run_path_posterior_flow.py
scripts/validate_benchmark_manifest.py
scripts/check_baselines_sanity.py
scripts/run_external_baseline_matrix.py
```

Deprecated diagnostic entry points:

```text
semflow_sr/train/train_block_flow.py
scripts/run_block_risk_flow.py
```

Smoke results under `results/semantic_fisher_flow_smoke/` are connectivity
checks, not final benchmark results.
