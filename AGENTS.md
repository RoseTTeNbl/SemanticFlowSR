# Agent Handoff: SemanticFlowSR

This is the first file a new coding agent should read in this repository.

## Repository Root

Work from:

```text
/home/ywj/wyh/SFSR/SemanticFlowSR
```

Use the `semflow` conda environment unless the user says otherwise.

```bash
conda activate semflow
```

If the environment must be recreated:

```bash
conda create -n semflow python=3.11
conda activate semflow
pip install --index-url https://download.pytorch.org/whl/cu126 torch
pip install numpy scipy sympy pyyaml pandas scikit-learn tqdm einops pytest
pip install deap gplearn
pip install -e .
```

## Current Main Method

The current method is Conditional Semantic Edge Flow, CSEF.

CSEF performs teacher velocity matching on explicit categorical probability
shapes over the expression construction graph. A ground-truth expression is
compiled to a canonical construction path. GT-neighborhood perturbations provide
noisy contexts, while the target probability shape flows to the clean GT action
as a smoothed one-hot endpoint. The hidden network does not store that shape
directly; it predicts conditional velocities from the task, construction
prefix, register semantics, current probability state, and time.

The active geometry split is:

```text
CSEF-Fisher:    Fisher-Rao probability-shape teacher path
CSEF-Euclidean: Euclidean probability-coordinate ablation
```

Semantic information is used as network input and as a local output-semantic
calibration matrix for velocity errors. It does not change the Fisher-Rao
teacher path.

Do not modify `/home/ywj/wyh/SFSR/paper` unless the user explicitly asks for
paper edits.

## Main Files

| Purpose | File |
|---|---|
| Template and register metadata | `semflow_sr/edge_flow/template.py` |
| Conditional CSEF model and sampler | `semflow_sr/edge_flow/conditional.py` |
| Probability-shape teacher velocity and semantic calibration | `semflow_sr/edge_flow/semantic_teacher.py` |
| GT-neighborhood noisy-context sampler | `semflow_sr/edge_flow/gt_neighborhood.py` |
| Formula-to-CSEF compiler | `semflow_sr/edge_flow/path_compiler.py` |
| Structure similarity and endpoint evidence helpers | `semflow_sr/edge_flow/structure_posterior.py` |
| Reward evaluation and sparse head fitting | `semflow_sr/edge_flow/reward.py` |
| Structure-prior rerank score | `semflow_sr/edge_flow/selection.py` |
| Benchmark loading and result writing | `semflow_sr/edge_flow/benchmark.py` |
| Training CLI | `semflow_sr/edge_flow/train_edge_flow.py` |
| Evaluation CLI | `scripts/run_edge_flow.py` |
| Paper metrics bundle builder | `scripts/archive_paper_metrics.py` |

Useful docs:

- `README.md`: quick command entry points and current headline metrics.
- `docs/ALGORITHM.md`: current algorithm and mathematical derivation.
- `docs/ARCHITECTURE.md`: code layout and data flow.
- `docs/MATH.md`: compact mathematical reference.
- `docs/RESULTS.md`: current Fisher and Euclidean GPU results.
- `docs/IMPROVEMENT_NOTES.md`: current evidence, limitations, and next checks.

## Current Commands

Fisher probability-shape training:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n semflow \
  python -u -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.yaml
```

Euclidean ablation training:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n semflow \
  python -u -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/conditional_edge_flow_gt_sampler_teacher_path_euclidean_gpu.yaml
```

Fisher evaluation:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n semflow \
  python -u scripts/run_edge_flow.py \
  --ckpt checkpoints/teacher_path_geometry/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.pt \
  --out results/teacher_path_geometry_fisher_gpu \
  --tag teacher_path_geometry_fisher_gpu \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --manifest_root data/benchmark_suites \
  --manifest_suite nguyen constant livermore jin \
  --feynman_root data/materialized/feynman \
  --eval_samples 64 \
  --flow_steps 1 \
  --sampler_method policy \
  --head_fit_mode linear \
  --device cuda:1
```

Euclidean evaluation:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n semflow \
  python -u scripts/run_edge_flow.py \
  --ckpt checkpoints/teacher_path_geometry/conditional_edge_flow_gt_sampler_teacher_path_euclidean_gpu.pt \
  --out results/teacher_path_geometry_euclidean_gpu_20260623 \
  --tag teacher_path_geometry_euclidean_gpu_20260623 \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --manifest_root data/benchmark_suites \
  --manifest_suite nguyen constant livermore jin \
  --feynman_root data/materialized/feynman \
  --eval_samples 64 \
  --flow_steps 1 \
  --sampler_method policy \
  --head_fit_mode linear \
  --device cuda:1
```

Build comparison metrics and figures:

```bash
conda run --no-capture-output -n semflow python scripts/archive_paper_metrics.py \
  --out results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623 \
  --suite nguyen constant livermore jin \
  --bootstrap_samples 1000 \
  --method CSEF-Fisher SFSR sfsr_method samples_jsonl results/teacher_path_geometry_fisher_gpu/teacher_path_geometry_fisher_gpu_samples.jsonl \
  --method CSEF-Euclidean SFSR sfsr_method samples_jsonl results/teacher_path_geometry_euclidean_gpu_20260623/teacher_path_geometry_euclidean_gpu_20260623_samples.jsonl
```

Regression tests:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n semflow \
  pytest tests/test_edge_flow_core.py tests/test_edge_flow_training.py \
         tests/test_external_adapter_outputs.py tests/test_paper_metrics.py -q
```

Dataset validation:

```bash
conda run -n semflow python scripts/validate_benchmark_manifest.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --root data/benchmark_suites \
  --out results/dataset_validation \
  --fail-on-error
```

## Current Results

The current result directories are:

```text
results/teacher_path_geometry_fisher_gpu/
results/teacher_path_geometry_euclidean_gpu_20260623/
results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623/
```

Headline metrics:

```text
Fisher:    n_tasks 34, R2 mean 0.937383, solution rate 0.323529, skeleton accuracy 0.029412
Euclidean: n_tasks 34, R2 mean 0.940868, solution rate 0.294118, skeleton accuracy 0.0
```

These retained artifacts should be regenerated after changing the teacher target
or rerank weights.

## Editing Rules For This Repo

- Prefer `rg` for searches.
- Keep generated documentation aligned with the current CSEF Fisher/Euclidean
  mainline.
- Keep result documentation limited to the current result directories above.
- Do not run full training unless the user explicitly asks.
- Future result and reflection documents should be written in Chinese,
  especially `docs/RESULTS.md` and `docs/IMPROVEMENT_NOTES.md`.
- When changing current behavior or entry points, update `AGENTS.md`,
  `README.md`, `docs/ALGORITHM.md`, `docs/ARCHITECTURE.md`,
  `docs/MATH.md`, and `docs/RESULTS.md` as needed.
