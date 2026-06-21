# Architecture

The current mainline is:

```text
Edge-Parameterized Semantic Flow Matching
```

The previous action-level Semantic-Fisher implementation remains under
`semflow_sr/path_posterior/` as legacy diagnostic code. New work should use
`semflow_sr/edge_flow/`.

## Main Edge Flow Modules

| File | Responsibility |
|---|---|
| `semflow_sr/edge_flow/template.py` | Register-operator circuit template and edge-choice group metadata. |
| `semflow_sr/edge_flow/edge_distribution.py` | Mixture edge distribution `Theta`, per-group probabilities, sqrt coordinates. |
| `semflow_sr/edge_flow/circuit_sampler.py` | Stratified complete-DAG sampling from `Theta`. |
| `semflow_sr/edge_flow/reward.py` | Complete-expression reward with affine calibration and complexity cost. |
| `semflow_sr/edge_flow/projection.py` | Top-k elite projection to target edge distribution `Theta*`. |
| `semflow_sr/edge_flow/flow_teacher.py` | Product Fisher square-root teacher path. |
| `semflow_sr/edge_flow/dataset.py` | Edge flow training records and diagnostics. |
| `semflow_sr/edge_flow/model.py` | Data-conditioned edge velocity model and loss. |
| `semflow_sr/edge_flow/train_edge_flow.py` | Smoke training CLI. |
| `scripts/run_edge_flow.py` | Smoke inference/evaluation CLI. |

## Reused Shared Modules

| File | Use |
|---|---|
| `semflow_sr/sr/ast.py` | Expression AST and numeric evaluation. |
| `semflow_sr/sr/ops.py` | Primitive registry and protected operators. |
| `semflow_sr/sr/printer.py` | Expression rendering/simplification for top candidates. |
| `semflow_sr/data/synthetic_generator.py` | Synthetic smoke tasks. |
| `semflow_sr/eval/metrics.py` | R2/NMSE conventions where needed. |

## Current Workflow

```text
configs/train/edge_flow_smoke.yaml
-> python -m semflow_sr.edge_flow.train_edge_flow
-> checkpoints/edge_flow_smoke.pt
-> scripts/run_edge_flow.py
-> results/edge_flow_smoke/
```

The smoke result files are intentionally small and tracked:

```text
results/edge_flow_smoke/edge_flow_smoke_summary.json
results/edge_flow_smoke/edge_flow_smoke_samples.jsonl
```

Future full benchmark outputs should go under a new dated directory and remain
ignored unless explicitly summarized.

## Legacy Code

Legacy action-level modules:

```text
semflow_sr/path_posterior/
semflow_sr/flow/semantic_fisher.py
semflow_sr/train/train_path_posterior_flow.py
scripts/run_path_posterior_flow.py
```

These are not deleted because they are useful for regression comparison, but
they should not be described as the current algorithm.
