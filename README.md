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
```

## Latest Verified Results

From `results/train_semantic_fisher.log`:

- final `semantic_fisher_velocity_loss`: `0.001397`
- held-out rollout `reward(r2)`: `0.9282`

From `results/semantic_fisher/formula_1var_seed0_summary.json` on 20 one-variable tasks:

- `mean R2`: `0.999041`
- `median R2`: `0.999999`
- `solution_rate`: `0.95`
- `steps_mean`: `4.6`

Local ranking diagnostics from the same run:

- `selected_reward_rank_mean`: `3.65`
- `pred_top1_reward_rank_mean`: `3.65`
- `selected_probability_rank_mean`: `1.0`
- `exact_semantic_fisher_top1_reward_rank_mean`: `6.4`
- `plain_fisher_top1_reward_rank_mean`: `15.4`

That is the current reason for the shift: the old plain Fisher-sphere line had
`solution_rate = 0.15` and `selected_reward_rank_mean = 23.625`; the semantic-Fisher
pullback version closes most of that gap.

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
