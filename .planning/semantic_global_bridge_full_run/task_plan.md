# Semantic Global Bridge Full Run

## Objective
Run one full train-test validation for the Global Bridge + Semantic Improvement Guidance algorithm, using task-level held-out evaluation and a separate results directory.

## Plan
- [complete] Inspect current implementation and command surface.
- [complete] Run static/smoke checks for train/eval split and bridge path.
- [complete] Launch full benchmark train-test run in a new validation directory.
- [complete] Monitor key losses, split guard, and generated result files.
- [complete] Summarize metrics and any runtime or algorithmic errors.

## Constraints
- Do not train and test on the same task IDs.
- Do not revive semantic_gradient/collocation_mixture/local tau teacher.
- Do not overwrite old clean_boundary or semantic_gradient results.
- Keep BFGS as evaluation/diagnostic only.
