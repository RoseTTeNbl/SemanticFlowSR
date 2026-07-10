# SemanticFlowSR Docs

Current active line:

```text
One-Step Semantic Fisher Cycle
```

The current runnable mainline is the `register_categorical_blocks`
construction branch. It samples source particles, proposes complete soft
endpoints, applies complete-expression semantic tilt, rebuilds a
source-preserving Fisher coupling, and trains both the Fisher velocity field
and the proposer from the same coupling:

```text
theta0 -> G_phi(theta0, D)
       -> complete-expression semantic tilt
       -> Fisher source-endpoint coupling
       -> V_psi(theta_t, theta0, D, t)
       -> same-coupling proposer update
```

Target-conditioned Stage1, semantic latent endpoint, token construction, and
Stage2 endpoint correction remain historical/failed probes. They should not be
used as current validation paths.

Read in this order:

```text
ALGORITHM_COMPLETE_EXPRESSION_SEMANTIC_FM.md
MATH.md
ARCHITECTURE_COMPLETE_EXPRESSION_SEMANTIC_FM.md
STRUCTURAL_CLOSURE.md
DIAGNOSTIC_EXPERIMENTS_COMPLETE_EXPRESSION_FLOW.md
```

Active scripts:

```text
scripts/train_complete_expression_semantic_fm.py
scripts/run_one_step_semantic_fisher_cycle_gpu.sh
```

Current validation artifacts:

```text
results/clean_benchmark_20260701/ablations/complete_expression_semantic_fm_20260707/
```

Preserved comparison baseline:

```text
results/clean_benchmark_20260701/ablations/clean_boundary_20260702/
```

Do not reintroduce as current mainline:

```text
target_conditioned_reference Stage1
semantic_latent_endpoint family mass
semantic_gradient local teacher
collocation_mixture state sampler
local tau_b semantic interpolation
group-sampling semantic_improvement_stage
endpoint_shape / denoising parameterization
terminal anchor/prior training objective
contrastive auxiliary loss
fixed expression pool distribution distillation
inactive prior supervision
```
