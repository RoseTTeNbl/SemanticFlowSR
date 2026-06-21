# Edge Flow 87-Task Evaluation Adaptation

This note records the required wiring for a future 87-task evaluation under the
new Edge-Parameterized Semantic Flow Matching mainline. It is a workflow note
only; no full 87-task training or benchmark run has been launched for this
implementation pass.

## Task Coverage

The existing 87-task suite combines:

| Source | Count |
|---|---:|
| Nguyen / Constant / Livermore / Jin formula tasks | 34 |
| Materialized Feynman tasks | 53 |
| Total | 87 |

The edge-flow template must be constructed per task input dimension. Unlike the
old fixed-`d` action model, the new runner should either build a template from
`task.X_train.shape[1]` at inference time or load a checkpoint whose template
metadata matches that dimension.

## Future Full Evaluation Command Shape

The smoke runner already accepts template metadata from the checkpoint. A full
runner should extend it with benchmark manifest loading:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python scripts/run_edge_flow.py \
  --ckpt checkpoints/edge_flow_<date>.pt \
  --out results/edge_flow_<date> \
  --tag edge_flow_87 \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --manifest_suite nguyen constant livermore jin \
  --legacy_87 \
  --eval_samples 4096 \
  --flow_steps 16 \
  --seed 0
```

The current `scripts/run_edge_flow.py` is intentionally smoke-sized and uses
synthetic tasks. Benchmark manifest support is the next evaluation task.

## Required Result Files

Future full runs should write:

```text
edge_flow_87_summary.json
edge_flow_87_samples.jsonl
edge_flow_87_metrics.csv
edge_flow_87_diagnostics.json
task_expressions.csv
statistics_by_group.csv
```

Per-task records should include:

| Field | Meaning |
|---|---|
| `ground_truth` | GT expression when available |
| `expression` | selected generated expression |
| `r2` / `nmse` / `reward` | final validation metrics |
| `complexity` | active DAG complexity |
| `best_train_r2` / `best_val_r2` | complete-expression reward diagnostics |
| `valid_expression_fraction` | sampled expression validity |
| `unique_expression_fraction` | duplicate rate diagnostic |
| `target_edge_entropy_mean` | projection concentration |
| `mode_entropy` | mixture mode collapse diagnostic |
| `flow_endpoint_distance` | final distance to learned endpoint proxy |

Group statistics should be reported for:

```text
all tasks
87-task suite
Jin subset
by number of input variables
by benchmark family
```

## Current Limitation

The smoke implementation validates the new data structure and training loop,
but it does not yet include benchmark-manifest evaluation, beam decoding, replay
elites, or mixture-mode training beyond the mixture-compatible data format.
