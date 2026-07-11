# SemanticFlowSR v5 architecture

## Public velocity interface

```text
TaskConditionedVelocityNetV5.forward(x, y, theta, t) -> tangent
```

The returned probability tangent is blockwise centered.  The interface does
not accept `theta0`, route metadata, or a particle identifier.

## Components

- `TaskConditionedVelocityNetV5`: shared task/state trunk and base direct field.
- `ConditionalSemanticPotentialV5`: small scalar potential used only while
  estimating the empirical weighted-Poisson correction.
- `ResidualVelocityHeadV5`: lightweight stage head trained on bridge residuals.
- `PoissonResidualVelocityV5`: frozen base field plus an ordered sum of frozen
  residual heads.  Task/state features are computed once per call.
- `semflow_sr.flow.semantic_poisson`: Fisher natural gradient, weak-Poisson
  objective, exponential correction, and finite residual utilities.

## Runtime path

```text
theta0
  -> RK2 rollout of v_k(theta,D,t)
  -> one hard trace and endpoint semantic energy
  -> task Poisson potential
  -> same-particle Fisher correction theta1_plus
  -> corrected analytic bridge
  -> residual-head update
```

Hard-prefix and signed-pair expression execution are endpoint-only operations.
They are not evaluated inside RK2.

## Checkpoint contract

The objective id is `semantic_poisson_residual_fisher_v5`.  Checkpoints record
the residual stage, correction step, Poisson configuration, and
`theta0_conditioning=none`.  A v2/v3/v4 checkpoint is rejected for v5 resume;
legacy evaluation remains a separate explicit path.

## Main entry points

- `scripts/train_complete_expression_semantic_fm.py`
- `scripts/run_one_step_semantic_fisher_cycle_gpu.sh`
- `scripts/check_external_baseline_preflight.py`
- `scripts/check_external_baseline_results.py`
