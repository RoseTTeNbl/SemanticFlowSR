# SemanticFlowSR v5 algorithm

The active method is `semantic_poisson_residual_fisher_v5_1_bootstrap_first`.  It transports a
population with one shared conditional velocity field

```text
v(theta, D, t),   D = (X, y),
```

and never conditions the network on `theta0`, a route id, or a particle id.
`theta0` is only the ODE initial state and an evaluation record.

## One outer iteration

1. Draw `K` independent source states and integrate the current field once for
   each source.  Each endpoint is hard-decoded into exactly one legal trace.
2. Fit affine coefficients on a deterministic 80% support split and score the
   expression on the remaining 20% query split.  The semantic energy is
   `E = 1 - R2_query`; invalid expressions receive a finite capped energy.
3. Fit a task-conditioned scalar potential with the centered empirical weak
   Poisson objective

   ```text
   mean_i[0.5 ||grad_FR phi_i||^2
          + (E_i - mean(E)) (phi_i - mean(phi))].
   ```

4. Correct each endpoint with its own Fisher natural-gradient step.  The source
   identity is unchanged: there is no assignment, resampling, Hungarian, or
   Sinkhorn stage.  A task with collapsed energy variance produces zero
   correction and is reported as `support_collapse`.
5. Build the corrected Fisher bridge from the same source and train only a
   lightweight residual head with

   ```text
   b_t = (u_t_plus - v_k(theta_t_plus, D, t)) / eta,
   v_{k+1} = v_k + eta * Delta_v.
   ```

The bootstrap stage compiles training GT traces only to initialize the base
field.  GT is absent from ordinary outer iterations and inference.

## Symbolic chart

The active chart has one readout.  Polynomial sums and differences are built by
register-level `add/sub` actions in canonical SSA order; CSE merges repeated
subexpressions.  ODE-time conditioning uses only smooth soft-register summaries.
Hard expression execution happens once per endpoint.

## Evaluation contract

Evaluation reports raw R2, support-fitted/query R2, skeleton match, operator
dependency match, and the full K-particle population.  Oracle-free selection is
the ordinary result; any GT best-of-K statistic must be labelled diagnostic.

## Removed active mechanisms

Mutation, elite/archive selection, multi-readout construction, capacity
resampling, matching, block-marginal projection, route-conditioned velocity,
and stacked empirical auxiliary losses are not part of v5.  Legacy checkpoints
can only be opened through the explicit legacy-evaluation path.
