# Results

Current active result family:

```text
Edge-Parameterized Semantic Flow Matching
```

Current tracked benchmark results:

```text
results/edge_flow_87_basic/
results/edge_flow_87_h4_l3_k8/
```

Each full result directory must contain:

```text
*_summary.json
*_samples.jsonl
*_task_expressions.csv
*_task_expressions.md
*_statistics_by_group.csv
*_statistics_by_group.json
*_diagnostics.json
```

The smoke run remains only a wiring check. It must not be compared to 87-task
results or external baselines.

## Smoke Result

Latest smoke command:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python scripts/run_edge_flow.py \
  --ckpt checkpoints/edge_flow_smoke.pt \
  --out results/edge_flow_smoke \
  --tag edge_flow_smoke \
  --num_tasks 2 \
  --eval_samples 64 \
  --flow_steps 4 \
  --seed 1
```

Current smoke summary:

```json
{
  "n_tasks": 2,
  "r2_mean": 0.8014429211616516,
  "solution_rate": 0.5
}
```

## 87-Task Basic Result

Run date: 2026-06-21.

Configuration:

```text
configs/train/edge_flow_87_basic.yaml
```

This is one low-budget Edge Flow configuration group. It trains one
`num_vars=3` checkpoint on the full 87-task train split and pads lower-variable
tasks with zero dummy columns.

Training:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/edge_flow_87_basic.yaml
```

Evaluation:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python scripts/run_edge_flow.py \
  --ckpt checkpoints/edge_flow_87_basic.pt \
  --out results/edge_flow_87_basic \
  --tag edge_flow_87_basic \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --manifest_root data/benchmark_suites \
  --manifest_suite nguyen constant livermore jin \
  --legacy_87 \
  --feynman_root data/materialized/feynman \
  --eval_samples 256 \
  --flow_steps 8 \
  --seed 0
```

Summary:

```json
{
  "n_tasks": 87,
  "r2_mean": 0.7931209405263265,
  "nmse_mean": 0.2068790586840885,
  "solution_rate": 0.06896551724137931,
  "skeleton_accuracy": 0.022988505747126436,
  "complexity_mean": 3.1264367816091956,
  "valid_expression_fraction_mean": 0.9987877155172413,
  "unique_expression_fraction_mean": 0.6968390804597702
}
```

`skeleton_accuracy` is a 0/1 exact operator-skeleton metric. It compares
`ground_truth` with the selected `raw_expression` after removing numeric
constants and affine scaling, stripping protected `Abs`, and preserving variable
identity plus discrete exponent structure. It is intentionally stricter than
R2: high R2 from a correlated but wrong structure does not count.

Output files:

```text
results/edge_flow_87_basic/edge_flow_87_basic_summary.json
results/edge_flow_87_basic/edge_flow_87_basic_samples.jsonl
results/edge_flow_87_basic/edge_flow_87_basic_task_expressions.csv
results/edge_flow_87_basic/edge_flow_87_basic_task_expressions.md
results/edge_flow_87_basic/edge_flow_87_basic_statistics_by_group.csv
results/edge_flow_87_basic/edge_flow_87_basic_statistics_by_group.json
results/edge_flow_87_basic/edge_flow_87_basic_diagnostics.json
```

Subset highlights:

| group | n | R2 mean | solution rate | skeleton acc. |
|---|---:|---:|---:|---:|
| all | 87 | 0.7931 | 0.0690 | 0.0230 |
| constant | 8 | 0.9492 | 0.1250 | 0.1250 |
| feynman | 53 | 0.7530 | 0.0566 | 0.0000 |
| jin | 6 | 0.5808 | 0.0000 | 0.0000 |
| livermore | 8 | 0.9021 | 0.1250 | 0.0000 |
| nguyen | 12 | 0.8998 | 0.0833 | 0.0833 |
| num_vars=1 | 21 | 0.9219 | 0.1429 | 0.0952 |
| num_vars=2 | 29 | 0.8196 | 0.1034 | 0.0000 |
| num_vars=3 | 37 | 0.6993 | 0.0000 | 0.0000 |

## 87-Task H4/L3/K8 Diagnostic Result

Run date: 2026-06-21.

Configuration:

```text
configs/train/edge_flow_87_h4_l3_k8.yaml
```

This run applies the first structural corrections from the new Edge Flow
diagnosis:

```text
H = 4 mixture modes
L = 3 template layers
K = 8 registers
per-mode top-k hard projection
validation-robust target rewards with validation_fraction = 0.25
decoder budget curve at 256 / 1024 / 4096 samples
prior and theta_star projection oracle diagnostics
```

Training:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/edge_flow_87_h4_l3_k8.yaml
```

Evaluation:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python scripts/run_edge_flow.py \
  --ckpt checkpoints/edge_flow_87_h4_l3_k8.pt \
  --out results/edge_flow_87_h4_l3_k8 \
  --tag edge_flow_87_h4_l3_k8 \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --manifest_root data/benchmark_suites \
  --manifest_suite nguyen constant livermore jin \
  --legacy_87 \
  --feynman_root data/materialized/feynman \
  --eval_samples 1024 \
  --flow_steps 8 \
  --seed 0 \
  --decoder_budgets 256 1024 4096 \
  --oracle_samples 2048 \
  --oracle_decode_samples 2048 \
  --elite_k 8 \
  --target_smoothing 0.01 \
  --projection_mode per_mode_topk \
  --selection_eta_logprob 0.0
```

Summary:

```json
{
  "n_tasks": 87,
  "r2_mean": 0.8184552048814708,
  "nmse_mean": 0.18154479791512626,
  "solution_rate": 0.09195402298850575,
  "skeleton_accuracy": 0.022988505747126436,
  "complexity_mean": 4.045977011494253,
  "valid_expression_fraction_mean": 0.9975866558908046,
  "unique_expression_fraction_mean": 0.6270878232758621
}
```

Comparison to `edge_flow_87_basic`:

| run | R2 mean | solution rate | skeleton acc. | complexity | valid frac. | unique frac. |
|---|---:|---:|---:|---:|---:|---:|
| basic H1/L2/K5 | 0.7931 | 0.0690 | 0.0230 | 3.1264 | 0.9988 | 0.6968 |
| H4/L3/K8 | 0.8185 | 0.0920 | 0.0230 | 4.0460 | 0.9976 | 0.6271 |

Subset highlights:

| group | n | R2 mean | solution rate | skeleton acc. |
|---|---:|---:|---:|---:|
| all | 87 | 0.8185 | 0.0920 | 0.0230 |
| constant | 8 | 0.9549 | 0.1250 | 0.1250 |
| feynman | 53 | 0.7666 | 0.0943 | 0.0000 |
| jin | 6 | 0.6840 | 0.0000 | 0.0000 |
| livermore | 8 | 0.9412 | 0.1250 | 0.0000 |
| nguyen | 12 | 0.9419 | 0.0833 | 0.0833 |
| num_vars=1 | 21 | 0.9551 | 0.1429 | 0.0952 |
| num_vars=2 | 29 | 0.8684 | 0.1724 | 0.0000 |
| num_vars=3 | 37 | 0.7018 | 0.0000 | 0.0000 |

Module-wise diagnostics:

| diagnostic | value |
|---|---:|
| prior oracle best R2 mean | 0.8491 |
| theta_star decode best R2 mean | 0.8743 |
| theta_star projection drop mean | -0.0252 |
| prior skeleton accuracy | 0.0230 |
| theta_star skeleton accuracy | 0.0345 |
| model skeleton accuracy | 0.0230 |
| decoder 256 R2 mean | 0.7707 |
| decoder 1024 R2 mean | 0.8273 |
| decoder 4096 R2 mean | 0.8678 |
| decoder 4096 skeleton accuracy | 0.0230 |
| median calibration gain | 1.4676 |
| tasks with calibration gain > 0.1 | 0.8276 |
| mean used variable count | 1.6092 |
| 3-variable full-dependency fraction | 0.0270 |

Interpretation:

```text
H4/L3/K8 improves numeric R2 and solution rate, and the decoder budget curve
shows that more samples recover better semantic fits. It does not improve
skeleton recovery. Prior and theta_star oracles also have high R2 but nearly
unchanged skeleton accuracy, so the main bottleneck is not only neural flow
imitation. The sampled target family and reward still prefer correlated proxy
expressions.
```

## Result Layout Going Forward

Use:

```text
results/edge_flow_smoke/          small wiring checks
results/edge_flow_87_<tag>/       full 87-task Edge Flow runs
results/edge_flow_<date>/         ignored scratch or slice experiments
```

For future full experiments, write:

```text
*_summary.json
*_samples.jsonl
*_metrics.csv
*_diagnostics.json
task_expressions.csv/md
statistics_by_group.csv/json
```

Only curated summaries should be tracked unless the user explicitly asks to
track raw benchmark outputs.
