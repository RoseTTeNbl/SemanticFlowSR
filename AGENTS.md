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

`deap` and `gplearn` are installed for GP-assisted SFSR candidate-pool and
likelihood work. They are not automatically the direct paper baselines.

## Current Main Method

The current main method is:

```text
Semantic-Fisher Flow Matching
```

The old H3/HxA RiskFlow path is deprecated as a mainline method. It remains in
the tree for diagnostics and regression comparison. The old
PathPosterior-Frequency action target has been removed from this branch.

The current implementation uses one executable action as the local decision
unit:

```text
omega = action
```

The action endpoint is:

```text
q_hat_s = TargetSampler(s, A_s, p_init)
```

where the target sampler produces only an empirical probability shape over the
current action support. Semantic effects are used by the geometry, not by target
construction.

Use `Semantic-Fisher Flow Matching` as the algorithm name. TargetSampler names
identify experiment settings, not separate algorithms.

Current TargetSampler settings:

| Experiment setting | Config `target_mode` | TargetSampler | Meaning |
|---|---|---|---|
| `OneStepTarget` | `one_step` | `OneStepTargetSampler` | Old dense one-step residual gain `E(B_s)-E(B_s^a)` converted to `q_hat` by rank-softmax. Keep as sanity/regression baseline. |
| `FutureGroup-L3Target` | `future_group_l3` | `FutureGroupTargetSampler` | Execute candidate action, sample length-3 short continuations, top-k mean rollout reward, then rank-softmax to `q_hat`. This is the current formal experiment group. |
| `CachedTrajectoryFitnessTarget` | `cached_trajectory_fitness` | `CachedTrajectoryFitnessTargetSampler` | Load cached trajectory records and convert fitness-weighted supported first actions into an empirical `q_hat`. |
| `GPCandidateFitnessTarget` | `gp_candidate_fitness` | `GPCandidateFitnessTargetSampler` | Load a trained GP population, combine computable likelihood terms with fitness, and sample a simplex point `q_hat`. |
| `ImportanceSamplingTarget` | `importance_sampling` | `ShapeSamplingTargetSampler` | Use one-step scores as target density and `p_init` as proposal; self-normalized samples define `q_hat`. |
| `MCMCShapeTarget` | `mcmc_shape` | `ShapeSamplingTargetSampler` | Run a small Metropolis chain on the support target density and return the empirical probability shape. |

Every TargetSampler returns a full probability shape `q_hat` over the support.
It does not choose or commit an action.

## Algorithm Chain

```text
collect prefix states from behavior trajectories
-> build deterministic p_init over the current action support
-> target sampler builds q_hat(action | state)
-> build lambda-dependent log-ratio log q_hat - log p_lambda
-> compute exact action residual-effect vectors xi(action)
-> build semantic Gram K(action, action')
-> solve semantic-Fisher teacher log-rate w*
-> train SemanticTransformer with square-root flow matching
-> inference integrates predicted log-rate and commits one action or STOP
```

Conceptual split:

```text
target sampler supplies endpoint probability shapes
semantic-Fisher geometry supplies local probability-flow geometry
```

Do not describe `xi` or Gram matrices as reward estimators.

## What Is Not Mainline

Do not promote these as the current algorithm unless explicitly asked for a
diagnostic or ablation:

- one-step local residual reward as the main target;
- legacy rollout-fitness advantage;
- H3/H5 block reward;
- `block_reward = max/topk/mean over suffixes`;
- factorized HxA coordinate target;
- zeta-averaged geometry;
- separate FullSelector / full-expression reranker;
- operator-level GP online bias;
- GP posterior correction as a default main-method weighting;
- risk-weighted visited-action frequency as the main target;
- terminal semantic projection as the main target;
- uniform exploration mixtures such as `(1-eps) policy + eps uniform`.

More exploration should be more complete trajectories from the recorded behavior
policy, or a separately labeled off-policy source with behavior probabilities.

## Main Files

| Purpose | File |
|---|---|
| Prefix trajectory records | `semflow_sr/path_posterior/target.py` |
| Target samplers and deterministic p_init | `semflow_sr/path_posterior/target_sampler.py` |
| STOP/support/health helpers | `semflow_sr/path_posterior/action_support.py` |
| Complete action trajectory sampler | `semflow_sr/path_posterior/sampler.py` |
| Dataset and semantic-Fisher target builder | `semflow_sr/path_posterior/dataset.py` |
| Action-flow trainer | `semflow_sr/train/train_path_posterior_flow.py` |
| Action-flow evaluation CLI | `scripts/run_path_posterior_flow.py` |
| Main model | `semflow_sr/models/semantic_transformer.py` |
| Action semantic effects | `semflow_sr/semantics/energy.py` |
| Semantic-Fisher action solver | `semflow_sr/flow/semantic_fisher.py` |
| Deprecated block diagnostics | `semflow_sr/blocks/`, `semflow_sr/train/train_block_flow.py` |
| Manifest validation | `scripts/validate_benchmark_manifest.py` |
| External baseline matrix planning | `scripts/run_external_baseline_matrix.py` |

Useful docs:

- `docs/ALGORITHM.md`: current algorithm.
- `docs/ARCHITECTURE.md`: code layout.
- `docs/IMPROVEMENT_NOTES.md`: why H3/HxA underperformed and what changed.
- `README.md`: quick commands.

## Commands

Targeted tests:

```bash
conda run -n semflow pytest -q tests/test_path_posterior_flow.py
```

Train the current 87-task action-flow checkpoint:

```bash
conda run -n semflow python -m semflow_sr.train.train_path_posterior_flow \
  --config configs/train/semantic_fisher_flow_87_future_group_l3.yaml
```

Evaluate the current action-flow checkpoint:

```bash
conda run -n semflow python scripts/run_path_posterior_flow.py \
  --ckpt checkpoints/semantic_fisher_flow_future_group_l3_87.pt \
  --legacy_87 \
  --out results/semantic_fisher_flow_87 \
  --tag semantic_fisher_flow_future_group_l3_87_seed0 \
  --max_steps 6 \
  --device cpu
```

Validate datasets:

```bash
conda run -n semflow python scripts/validate_benchmark_manifest.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --root data/benchmark_suites \
  --out results/dataset_validation \
  --fail-on-error
```

Build external-baseline command plans:

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --suite_group formula_dev srsd_main srsd_dummy pmlb \
  --plan_out results/benchmark_plans/external_baseline_commands.json
```

## Smoke Versus Full Experiment

`smoke` means connectivity check. It answers:

```text
Does the path run end to end without crashing?
```

It does not answer:

```text
Is the algorithm competitive?
```

Current completed 87-task action-flow runs:

| TargetSampler | Checkpoint | Result summary | Mean R2 | Median R2 | Solution rate | STOP task fraction |
|---|---|---|---:|---:|---:|---:|
| `OneStepTarget` | `checkpoints/semantic_fisher_flow_one_step_87.pt` | `results/semantic_fisher_flow_87/semantic_fisher_flow_one_step_87_seed0_summary.json` | 0.8130 | 0.9036 | 0.1494 | 0.1034 |
| `FutureGroup-L3Target` | `checkpoints/semantic_fisher_flow_future_group_l3_87.pt` | `results/semantic_fisher_flow_87/semantic_fisher_flow_future_group_l3_87_seed0_summary.json` | 0.8466 | 0.9122 | 0.1149 | 0.0575 |

These are not paper results; treat them as current 87-task pilots for
Semantic-Fisher Flow Matching under the listed TargetSampler settings.

Before a full experiment, verify:

- complete trajectories are sampled from the recorded behavior model;
- no uniform exploration mixture is hidden in the sampler;
- lower-dimensional tasks are padded to checkpoint `d`;
- higher-dimensional tasks are skipped or assigned a matching checkpoint bucket;
- oracle / exact teacher / learned rollout are reported separately;
- baseline adapters pass sanity tasks before formal comparison.

## Dataset and Checkpoint Rules

The benchmark manifest is:

```text
data/benchmark_suites/benchmark_manifest.json
```

The action-flow evaluator pads tasks with fewer variables than the checkpoint
model dimension. Tasks with more variables than the checkpoint supports are
skipped and logged.

For generated programs, checkpoint register capacity must still be large enough
for variables, constants, and committed actions:

```text
K >= num_vars + 1 + max_steps
```

## Editing Rules For This Repo

- Prefer `rg` for searches.
- Do not delete legacy code just because it is not mainline; tests and archived
  scripts may still rely on it.
- Do not reintroduce local residual reward into the main path-posterior target.
- Do not add hidden exploration mixtures to the sampler.
- When changing behavior or entry points, update `README.md`, `AGENTS.md`,
  `docs/ALGORITHM.md`, and `docs/ARCHITECTURE.md`.
