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

The current main method is:

```text
Edge-Parameterized Semantic Flow Matching
```

Do not describe the current algorithm as action-level SFFM/TSSF. The old
`path_posterior` implementation is legacy diagnostic code.

Current probability object:

```text
Theta = mixture edge distribution over register-operator circuit templates
```

Current algorithm chain:

```text
data D=(X,y)
-> build RegisterOperatorTemplate
-> initialize uniform EdgeDistribution Theta0
-> sample complete expression DAGs from q_Theta
-> evaluate complete-expression rewards
-> project top-k elites to Theta*
-> build product Fisher square-root teacher path
-> train EdgeFlowModel on dot z
-> inference integrates Theta and samples complete expressions
```

## Main Files

| Purpose | File |
|---|---|
| Template and edge groups | `semflow_sr/edge_flow/template.py` |
| Edge distribution | `semflow_sr/edge_flow/edge_distribution.py` |
| Complete DAG sampler | `semflow_sr/edge_flow/circuit_sampler.py` |
| Reward evaluator | `semflow_sr/edge_flow/reward.py` |
| Elite projection | `semflow_sr/edge_flow/projection.py` |
| Fisher-slerp teacher | `semflow_sr/edge_flow/flow_teacher.py` |
| Training records | `semflow_sr/edge_flow/dataset.py` |
| Edge-flow model/loss | `semflow_sr/edge_flow/model.py` |
| Smoke trainer | `semflow_sr/edge_flow/train_edge_flow.py` |
| Smoke evaluator | `scripts/run_edge_flow.py` |
| Smoke config | `configs/train/edge_flow_smoke.yaml` |

Useful docs:

- `docs/ALGORITHM.md`: current algorithm.
- `docs/ARCHITECTURE.md`: code layout.
- `docs/IMPROVEMENT_NOTES.md`: current bottlenecks and next milestones.
- `docs/RESULTS.md`: current smoke-only result.

## Commands

Targeted tests:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  pytest -q tests/test_edge_flow_core.py tests/test_edge_flow_training.py
```

Smoke train:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/edge_flow_smoke.yaml
```

Smoke evaluate:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python scripts/run_edge_flow.py \
  --ckpt checkpoints/edge_flow_smoke.pt \
  --out results/edge_flow_smoke \
  --tag edge_flow_smoke
```

Validate datasets:

```bash
conda run -n semflow python scripts/validate_benchmark_manifest.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --root data/benchmark_suites \
  --out results/dataset_validation \
  --fail-on-error
```

## Smoke Versus Full Experiment

`smoke` means connectivity check:

```text
Does the Edge Flow path run end to end without crashing?
```

It does not answer:

```text
Is the algorithm competitive on 87 tasks?
```

Current tracked smoke output:

```text
results/edge_flow_smoke/
```

## Legacy Code

Do not delete legacy code casually; tests and archived comparisons may rely on
it. But do not promote it as current:

```text
semflow_sr/path_posterior/
semflow_sr/flow/semantic_fisher.py
semflow_sr/train/train_path_posterior_flow.py
scripts/run_path_posterior_flow.py
```

## Editing Rules For This Repo

- Prefer `rg` for searches.
- Keep CPU-heavy commands capped with `taskset -c 0-3` and thread env vars.
- Do not run full training unless the user explicitly asks.
- When changing current behavior or entry points, update `AGENTS.md`,
  `docs/ALGORITHM.md`, `docs/ARCHITECTURE.md`, and `docs/RESULTS.md`.
