# SemanticFlowSR

SemanticFlowSR is currently centered on:

```text
Semantic-Fisher Flow Matching
```

The mainline is not the older H3/HxA RiskFlow path. For each state, a
deterministic initial probability shape is built over the current action
support, then a TargetSampler constructs an empirical endpoint distribution:

```text
q_hat(omega | s)
```

The current implementation uses:

```text
omega = one executable action
```

The algorithm name stays `Semantic-Fisher Flow Matching`; experiment groups are
named by their TargetSampler. The current formal group is
`FutureGroup-L3Target`: each candidate action is scored by short rollout groups
of depth 3, and those scores induce `q_hat`. Semantic effects define only the
semantic-Fisher geometry.

For a new coding session, read [AGENTS.md](AGENTS.md), then
[docs/ALGORITHM.md](docs/ALGORITHM.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Current Algorithm

```text
sample complete action trajectories from the behavior model
-> collect prefix states
-> build deterministic p_init over current action support
-> target sampler builds q_hat(action | state)
-> build lambda-dependent A_lambda = log q_hat - log p_lambda
-> compute exact local action residual effects xi(action)
-> solve semantic-Fisher teacher log-rate w*
-> train SemanticTransformer by square-root flow matching
-> inference integrates predicted velocity and commits one action or STOP
```

## TargetSampler Groups

| Experiment setting | Config `target_mode` | TargetSampler | Endpoint construction |
|---|---|---|---|
| `OneStepTarget` | `one_step` | `OneStepTargetSampler` | Scores each action by the old dense one-step residual gain `E(B_s)-E(B_s^a)`, then rank-softmaxes scores into `q_hat`. This is a sanity/regression baseline. |
| `FutureGroup-L3Target` | `future_group_l3` | `FutureGroupTargetSampler` | Executes each candidate action, samples short continuations of length `L=3`, aggregates rollout rewards by top-k mean, then rank-softmaxes scores into `q_hat`. This is the current formal experiment group. |
| `CachedTrajectoryFitnessTarget` | `cached_trajectory_fitness` | `CachedTrajectoryFitnessTargetSampler` | Loads cached trajectory records, maps supported first actions to fitness-weighted trajectory samples, and returns an empirical `q_hat`. |
| `GPCandidateFitnessTarget` | `gp_candidate_fitness` | `GPCandidateFitnessTargetSampler` | Loads a trained GP population or trajectory records, combines computable GP likelihood terms with fitness, samples a point on the action-support simplex, and returns that `q_hat`. |
| `ImportanceSamplingTarget` | `importance_sampling` | `ShapeSamplingTargetSampler` | Uses one-step scores as the target density and `p_init` as proposal; self-normalized importance samples produce `q_hat`. |
| `MCMCShapeTarget` | `mcmc_shape` | `ShapeSamplingTargetSampler` | Runs a small Metropolis chain over the support target density and returns the empirical endpoint probability shape `q_hat`. |

Removed historical targets are not active experiment groups: risk-weighted
visited-action frequency and terminal semantic projection.

All TargetSampler variants output a full probability shape over the current
support. They do not return a sampled action; inference still integrates the
learned Semantic-Fisher velocity and then commits an action.

Important boundaries:

- No local one-step residual reward in the main target.
- No risk-weighted visited-action frequency as the main target.
- No terminal semantic projection as the main target.
- No `block_reward = max/topk/mean over suffixes`.
- No HxA coordinate target or zeta-averaged geometry as mainline.
- No uniform exploration mixture such as `(1-eps) policy + eps uniform`.
- More exploration means more complete trajectories from the recorded behavior
  policy, or a separately labeled off-policy source.
- GP/replay samples are not on-policy training data unless behavior
  probabilities and correction are recorded.
- `deap` and `gplearn` in the main env are for GP-assisted SFSR tools; external
  paper baselines are handled by the separate baseline scripts.

## Environment

Run commands from `SemanticFlowSR/`.

```bash
conda create -n semflow python=3.11
conda activate semflow

# Choose the PyTorch wheel matching the machine. This repo has been run with cu126.
pip install --index-url https://download.pytorch.org/whl/cu126 torch

pip install numpy scipy sympy pyyaml pandas scikit-learn tqdm einops pytest
pip install deap gplearn
pip install -e .
```

## Quick Commands

```bash
# Unit tests for the current Semantic-Fisher Flow Matching path.
conda run -n semflow pytest -q tests/test_path_posterior_flow.py

# Train the current 87-task action-flow checkpoint.
conda run -n semflow python -m semflow_sr.train.train_path_posterior_flow \
  --config configs/train/semantic_fisher_flow_87_future_group_l3.yaml

# Evaluate the current action-flow checkpoint on the legacy 87-task loader.
conda run -n semflow python scripts/run_path_posterior_flow.py \
  --ckpt checkpoints/semantic_fisher_flow_future_group_l3_87.pt \
  --legacy_87 \
  --out results/semantic_fisher_flow_87 \
  --tag semantic_fisher_flow_future_group_l3_87_seed0 \
  --max_steps 6 \
  --device cpu

# Validate the unified benchmark manifest before full experiments.
conda run -n semflow python scripts/validate_benchmark_manifest.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --root data/benchmark_suites \
  --out results/dataset_validation \
  --fail-on-error
```

The old block-flow commands still exist for diagnostics:

```text
semflow_sr/train/train_block_flow.py
scripts/run_block_risk_flow.py
```

Do not treat them as the current algorithm unless explicitly running a
comparison against the deprecated H3/HxA path.

## Main Code Map

| Purpose | File |
|---|---|
| Prefix trajectory records | `semflow_sr/path_posterior/target.py` |
| Target samplers and p_init | `semflow_sr/path_posterior/target_sampler.py` |
| STOP/support/health helpers | `semflow_sr/path_posterior/action_support.py` |
| Complete action trajectory sampler | `semflow_sr/path_posterior/sampler.py` |
| Dataset and teacher target builder | `semflow_sr/path_posterior/dataset.py` |
| Action-flow trainer | `semflow_sr/train/train_path_posterior_flow.py` |
| Action-flow evaluator | `scripts/run_path_posterior_flow.py` |
| Core action-support model | `semflow_sr/models/semantic_transformer.py` |
| Semantic-Fisher action solver | `semflow_sr/flow/semantic_fisher.py` |
| Deprecated H3/HxA diagnostic path | `semflow_sr/blocks/`, `semflow_sr/flow/semantic_fisher_table.py` |
| Dataset manifest tooling | `semflow_sr/data/benchmark_manifest.py`, `scripts/prepare_benchmark_suites.py` |
| External baseline tooling | `scripts/check_baselines_sanity.py`, `scripts/run_external_baseline_matrix.py` |
| GP-assisted SFSR helpers | `semflow_sr/gp_distill/` |

## Benchmark Layout

The unified manifest is:

```text
data/benchmark_suites/benchmark_manifest.json
```

Target suites:

```text
formula-dev: Nguyen / Constant / Livermore / Jin
SRSD-Feynman: easy / medium / hard
SRSD-Feynman dummy-variable variants
PMLB/SRBench filtered regression subset
```

Checkpoint dimension rules still matter. A task with more variables than the
checkpoint model supports must be skipped or evaluated with a matching
checkpoint bucket. The action-flow evaluator pads lower-dimensional tasks to
the checkpoint `d` and skips higher-dimensional tasks.

## Current Status

Current Semantic-Fisher Flow Matching 87-task runs:

| TargetSampler | Checkpoint | Result summary | Mean R2 | Median R2 | Solution rate | STOP task fraction |
|---|---|---|---:|---:|---:|---:|
| `OneStepTarget` | `checkpoints/semantic_fisher_flow_one_step_87.pt` | `results/semantic_fisher_flow_87/semantic_fisher_flow_one_step_87_seed0_summary.json` | 0.8130 | 0.9036 | 0.1494 | 0.1034 |
| `FutureGroup-L3Target` | `checkpoints/semantic_fisher_flow_future_group_l3_87.pt` | `results/semantic_fisher_flow_87/semantic_fisher_flow_future_group_l3_87_seed0_summary.json` | 0.8466 | 0.9122 | 0.1149 | 0.0575 |

Verification after the latest TargetSampler changes:

```text
tests/test_path_posterior_flow.py: 11 passed
```
