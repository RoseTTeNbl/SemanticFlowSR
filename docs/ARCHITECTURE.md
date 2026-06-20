# Architecture

This document maps the current Semantic-Fisher Flow Matching implementation.

Main flow:

```text
behavior trajectories
-> prefix states
-> deterministic support and p_init
-> TargetSampler endpoint q_hat
-> exact action semantic effects and Gram
-> lambda-dependent semantic-Fisher endpoint teacher
-> SemanticTransformer flow-matching loss
-> inference integrates predicted velocity and commits action or STOP
```

## Main Package

### `semflow_sr/path_posterior/action_support.py`

Defines STOP, STOP features/effects, health filtering, and support helpers. The
current efficient order is deterministic cap first, health filtering second,
STOP append last.

### `semflow_sr/path_posterior/target.py`

Contains only prefix collection records:

```text
PathDecision
PathTrajectory
```

The old PathPosterior-Frequency conditional code has been removed.

### `semflow_sr/path_posterior/target_sampler.py`

Defines target endpoint builders:

```text
PriorConfig
build_p_init
TargetShape
OneStepTargetSampler
FutureGroupTargetSampler
CachedTrajectoryFitnessTargetSampler
GPCandidateFitnessTargetSampler
ShapeSamplingTargetSampler
```

Target samplers produce `q_hat`, `target_scores`, and diagnostics. They do not
compute semantic effects or semantic-Fisher fields.

### `semflow_sr/path_posterior/sampler.py`

Samples behavior trajectories to collect prefix states. It records support and
the behavior policy at each visited state. Support construction now caps raw
actions before health filtering to avoid evaluating hundreds of actions before
the approximation cap.

### `semflow_sr/path_posterior/dataset.py`

Builds flow-matching records:

```text
synthetic task
-> behavior model samples root trajectories
-> collect selected prefix states
-> build p_init
-> target sampler builds q_hat
-> compute xi / Gram
-> integrate endpoint semantic-Fisher path
-> emit lambda-time records
```

Records keep compatibility with `collate_velocity`.

## Flow

### `semflow_sr/flow/semantic_fisher.py`

Important functions:

```text
semantic_fisher_lograte
semantic_fisher_sphere_step
integrate_semantic_fisher_endpoint_path
```

The endpoint path recomputes:

```text
log q_eps - log p_lambda
```

at each lambda-time state.

## Training

### `semflow_sr/train/train_path_posterior_flow.py`

Runs iterative behavior refresh and CPU-limited training. Config field:

```yaml
runtime:
  torch_num_threads: 4
  torch_num_interop_threads: 1
```

Checkpoint metadata uses algorithm names such as:

```text
semantic_fisher_flow_matching_future_group_l3
```

## Evaluation

### `scripts/run_path_posterior_flow.py`

Loads a checkpoint and uses the same support cap and deterministic `p_init`
STOP-bias rule as training. It does not call TargetSampler at inference time.

## Deprecated Diagnostics

The old block-flow and legacy endpoint paths remain for comparison:

```text
semflow_sr/blocks/
semflow_sr/flow/semantic_fisher_table.py
semflow_sr/models/block_flow_model.py
semflow_sr/train/train_block_flow.py
scripts/run_block_risk_flow.py
```
