# Fixed Symbol Node Stage 1 Findings

## Theory

- The new chart should use fixed symbol/operator nodes per layer.
- Probability blocks parameterize source edges for each node input slot plus final readout choices.
- Node outputs are written to fixed node positions for the next layer; other values are implicit carry-through via base inputs and previous node bank.
- Stage 1 should train an easy reference flow from random source simplexes to sharp syntax-valid trace endpoints.
- Semantic control and node-level semantic rewards remain future Stage 2 work.

## Current Implementation Notes

- `scripts/train_fixed_symbol_node_stage1.py` exists as a standalone prototype.
- It samples random valid traces, traces active ancestors from readout, and gives inactive blocks weight zero.
- It currently needs static/runtime validation.
- Initial fixed-batch run with independent `theta0` decreased from about 6.2 to 3.9 by epoch 16, but was too slow and not close to overfit convergence.
- Teacher scale diagnostic on 200 traces with `L=8`:
  zero-prediction loss mean about 6.33, target logit velocity max up to 66, probability velocity max about 1.44.
- The major theoretical issue is not only target scale: if `theta0` and endpoint are independent, low/intermediate `theta_t` often cannot identify which endpoint route generated it, so the Eulerian target becomes a high-variance route average.
- Stage 1 should use `theta0` as a route seed. Added `choice_bias` coupling so active endpoint choices get a small positive bias in the initial random logits.
- With `choice_bias=3.0`, active target choice is argmax at `theta_t` for about 98.8% of active blocks, but exact Fisher velocity still depends on the source route seed.
- Therefore Stage 1 should condition the velocity model on the known initial `theta0` seed. This makes the learned vector field a valid augmented-state conditional flow `V(theta_t, t, theta0)`, avoiding forced averaging over different source paths.

## Stage 2 Design

- Online guidance estimates a path-space posterior from complete trace samples:
  `q(z) proportional P_theta(z) exp(-Energy(z)/tau)`.
- Energy is computed only from complete decoded expressions:
  normalized MSE against target sampled semantics plus a small complexity penalty.
- The sampled posterior is projected to active block/action marginals only. Inactive sampled choices are ignored because the decoded expression does not depend on them.
- The probability-coordinate correction is `q_b - p_b`, scaled, time-gated, FR-capped, and converted to centered logit velocity before adding to the Stage 1 reference velocity.
- This is a rollout-time control estimate, not a new Stage 1 training loss.
