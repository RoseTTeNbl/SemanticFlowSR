# SemanticFlowSR

SemanticFlowSR now uses a single main algorithm:

```text
centered semantic energy
-> action semantic effect vectors
-> semantic-Fisher pullback metric
-> exact local log-rate target from a small linear system
-> square-root sphere update
-> model learns the local update operator
```

The main model output is not a free velocity and not a scalar endpoint potential. It
predicts a support-local semantic-Fisher `lograte` `w_theta(c, a)`. Training matches the
target square-root sphere tangent `z_dot`, and inference applies one positive sphere
retraction step.

Only two legacy comparisons remain: `gamma=0` inside the semantic-Fisher solver
(no pullback term), and the plain Fisher `closed_form` / `SpherePathLoss` potential
ablation.

## Environment

Run from `SemanticFlowSR/` in the `semflow` environment.

```bash
conda create -n semflow python=3.11
conda activate semflow
pip install --index-url https://download.pytorch.org/whl/cu126 torch
pip install numpy scipy sympy pyyaml pandas scikit-learn tqdm einops pytest
pip install -e .
```

## Verified Workflow

```bash
# 0. unit tests
conda run -n semflow pytest -q

# 1. train the base semantic-Fisher model
conda run -n semflow python -m semflow_sr.train.train_base_natural_flow \
  --config configs/train/base_natural_flow.yaml

# 2. evaluate the checkpoint with semantic-Fisher sphere updates
conda run -n semflow python scripts/run_experiment.py \
  --ckpt checkpoints/velocity_one_step_advantage.pt \
  --suite nguyen constant livermore jin \
  --seed 0 \
  --out results/semantic_fisher \
  --tag formula_1var_seed0 \
  --max_steps 12 \
  --grid 1 \
  --step_size 1.0 \
  --max_support 32 \
  --support_mode mixed_topk_random \
  --support_topk 16 \
  --target one_step_advantage \
  --integration_method semantic_fisher_sphere \
  --beta 1.0 \
  --gamma 0.1 \
  --record_diagnostics \
  --record_path \
  --device cpu

# 3. evaluate the future-reward ODE checkpoints on a filtered multivariate subset
conda run -n semflow python scripts/run_experiment.py \
  --ckpt_by_vars \
    1:checkpoints/velocity_rollout_future_ode_d1.pt \
    2:checkpoints/velocity_rollout_future_ode_d2.pt \
    3:checkpoints/velocity_rollout_future_ode_d3.pt \
  --suite nguyen constant livermore jin \
  --min_vars 2 \
  --require_all_ckpts \
  --out results/manual_future_ode_multivar \
  --tag multivar_seed0 \
  --support_mode adaptive_full \
  --support_full_threshold 256 \
  --max_support 64 \
  --support_topk 48 \
  --integration_method semantic_fisher_ode \
  --ode_steps 4 \
  --step_size 0.25 \
  --gram_rank 8 \
  --target rollout_fitness_advantage \
  --record_diagnostics

# 4. regenerate paper-facing aggregate tables and compact figures
conda run -n semflow python scripts/build_paper_results.py --out results/paper
```

## Latest Verified Results

Paper-facing 87-task benchmark results are under `results/paper/` and summarized in
`results/baseline_comparison.md`. The full benchmark composition is:

```text
Nguyen 12 + Constant 8 + Livermore 8 + Jin 6 + Feynman 53 = 87
variable counts: 1-var 21, 2-var 29, 3-var 37
```

Current 8-method paper table:

| Group | Method | Mean R2 | Solution rate |
|---|---|---:|---:|
| Baselines | PySR | 0.9974 | 0.9310 |
| Baselines | DEAP | 0.9455 | 0.3448 |
| Baselines | DSO | 0.9544 | 0.5977 |
| SFSR ablations | Ours one-step reward | 0.9708 | 0.7701 |
| SFSR ablations | Ours one-step ODE | 0.9505 | 0.7126 |
| SFSR main | Ours future ODE (no GP) | 0.9505 | 0.7126 |
| GP variants | GP as rollout policy | 0.9513 | 0.7011 |
| GP variants | GP policy distillation | 0.8846 | 0.3678 |

The earlier 14-task result was a multivariate-only built-in formula subset
(`Nguyen 4 + Constant 3 + Livermore 1 + Jin 6`) and is not directly comparable to the
87-task table.

## Current Entry Points

| Purpose | File |
|---|---|
| Base training entry | `semflow_sr/train/train_base_natural_flow.py` |
| Local semantic effects | `semflow_sr/semantics/energy.py` |
| Semantic-Fisher solver | `semflow_sr/flow/semantic_fisher.py` |
| Main trainer loss | `semflow_sr/train/losses.py::SemanticFisherVelocityLoss` |
| Main model | `semflow_sr/models/semantic_transformer.py` |
| Inference update helpers | `semflow_sr/inference/iterative_policy_update.py` |
| Search / rollout inference | `semflow_sr/search/rollout_velocity.py` |
| Benchmark evaluation | `scripts/run_experiment.py` |
| GP extension interfaces | `semflow_sr/gp_distill/`, `semflow_sr/targets/gp_implicit_target.py` |

## Layout

```text
configs/      experiment configs
semflow_sr/   core package
scripts/      data/eval/baseline CLIs
tests/        correctness and workflow tests
docs/         algorithm, theory mapping, architecture, dataset notes
results/      generated experiment reports
```

See [docs/README.md](docs/README.md) for the internal docs index.
