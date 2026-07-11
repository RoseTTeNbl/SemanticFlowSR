# One-Step Semantic Fisher Cycle

## Objective

Implement and validate v4 lineage-proximal single-expression semantic Fisher flow:

1. Bootstrap `v_psi` from strict GT traces, then use learned-flow rollout as the reference law.
2. Gate learned endpoints on distance to their own decoded single-expression cells.
3. Only after the gate passes, enumerate a small register-semantic-ranked local cell neighborhood.
4. Select the best raw semantic expression inside a Fisher radius without global sampling or KL tilt.
5. Preserve each source lineage; do not use cross-source Sinkhorn/Hungarian recoupling.
6. Train direct Fisher velocity matching with semantic conditioning, GT replay, inactive identity,
   and bounded terminal consistency.
7. Retract eval endpoints to the decoded single-expression stratum and report the projection cost.

Removed role:

- No one-step student/proposer is trained or used for endpoint generation/evaluation.

## Constraints

- Construction graph is `register_categorical_blocks`.
- No latent component matching or family-mass acceptance metric.
- No silent constant replacement, trace truncation, or inactive-block filling.
- Unsupported expressions fail compilation explicitly.
- Training and inference use the same block masks and endpoint decoding rules.
- Eval inference starts from random `theta0` and does not use compiled GT traces to construct endpoints; GT traces are diagnostics only unless an explicit oracle flag is enabled.
- Affine/post-fit metrics are reported separately and never replace raw metrics.
- Paper visualization uses fixed-source ODE snapshots with shared parameter/semantic embeddings;
  learned flow, local proximal targets, and diagnostic GT cells remain visually distinct.

## Phases

- [complete] Audit current endpoint and corrected-bridge baseline.
- [complete] Define particle, tilt, coupling, and cycle interfaces.
- [complete] Implement source-tracked endpoint atoms and flow matching.
- [complete] Add reliable diagnostics and trajectory visualization.
- [complete] Update algorithm, architecture, math, and closure docs.
- [in_progress] Remove discarded docs, results, and logs.
- [complete] Run focused tests and bounded validation for reference-bridge proposer.
- [complete] Verify normal eval does not mix reference oracle records into sample outputs.
- [complete] Implement flow-first semantic tilt with graph-native sharp targets.
- [complete] Add graph simplex visualization and flow-first diagnostics.
- [complete] Re-run unit, integration, smoke, and staged validation.
- [complete] Run medium GPU benchmark for several outer iterations.
- [in_progress] Audit the Fisher objective, block construction, ODE realization, and retained
  run metrics to localize the learned-endpoint failure before changing the algorithm.
- [pending] Add relative-to-zero and per-block-kind Fisher diagnostics, including readout
  terminal sharpness/margin trajectories and population-level eval summaries.
- [pending] Run focused tests and a short bounded GPU diagnostic; require evidence of readout
  sharpening and materially sub-baseline error before any longer run.
- [pending] Apply the smallest algorithmic correction justified by the diagnostic, then compare
  against an unchanged-control run with identical seeds and compute budget.
- [pending] Broaden evaluation across tasks and deterministic `theta0` populations; report
  distributional/common-error metrics rather than best-of-population results alone.
- [pending] Clean superseded diagnostic artifacts, update Chinese experiment/reflection records,
  and synchronize any major algorithm change to git without touching unrelated worktree changes.
- [complete] Implement v3 core probability tools: KL-constrained semantic weights, log-domain
  Sinkhorn transport, correction-budget diagnostics, and v3 objective versioning.
- [complete] Enforce the single-readout mainline and canonical full-expression register compilation.
- [complete] Replace heuristic cycle collection with learned rollout, complete-trace sampling,
  GT-anchor semantic proximal weighting, and soft reference-aware coupling.
- [complete] Enable soft/hard register semantic conditioning; add GT replay/teacher, inactive
  identity, differentiable terminal consistency, and eval terminal retraction.
- [complete] Add v3 unit/integration tests, relative/per-block diagnostics, and population eval
  summaries; run the focused regression suite.
- [complete] Run a bounded fixed-seed CPU/GPU diagnostic only; compare against the retained control
  before permitting medium training.
- [complete] Analyze the failed v3 smoke under the lineage-preserving semantic-proximal theory and
  reject KL-posterior/Sinkhorn recoupling as the active update.
- [complete] Replace the active outer update with per-lineage local Fisher-ball proximal MAP:
  learned endpoint, manifold gate, small deterministic cell neighborhood, raw semantic objective,
  same-source target bridge.
- [complete] Add direct rollout-to-GT diagnostics: hard GT hit, sampled GT hit, compiled GT trace mass,
  distance to decoded cell, and distance to nearest GT cell.
- [complete] Record fixed-source ODE snapshots across outer iterations and generate a paper-ready
  three-row parameter/expression 2D coupling figure for first/middle/final iterations.
- [complete] Update tests and run only focused/CPU smoke validation for the simplified proximal path;
  do not start another GPU or medium run because the existing smoke already fails the gates.
- [pending] Update Chinese algorithm/math/closure docs, clean only v3 superseded artifacts, and
  commit the major algorithm change without including unrelated worktree files.
- [in_progress] Fully reconstruct every organized external-baseline result directory and dataset by
  rerunning the repository's baseline scripts in their declared environments; preserve the running
  v4 medium output and suspend all code cleanup until baseline manifests/tables are restored.
- [pending] Monitor the medium bootstrap and first outer iteration; diagnose endpoint realization,
  literal GT generation, active manifold gap, runtime, and whether the local proximal gate opens.
- [pending] Remove inactive v2/v3 collection, KL/Sinkhorn, mutation/archive, obsolete proposer,
  online-guidance, and stale CLI/summary interfaces from the active trainer without changing the
  already-running medium process.
- [pending] Run focused regression/static checks after cleanup and write a concise handoff recording
  the launched run, source version boundary, diagnostics, retained interfaces, and remaining risks.

## Acceptance

- Strict compiler semantic oracle pass is explicit for every accepted trace.
- Coupling has uniform source occupancy and no dropped source particle.
- Semantic score records unfitted and coefficient-fitted errors.
- Flow tangent is finite, blockwise zero-sum, and trained on the same coupled pairs.
- No one-step proposer/student update is present in the active cycle.
- Outer iteration 2+ is proposed by the learned direct-velocity rollout; active v4 metrics report
  zero mutation, elite-selection, and previous-archive participation.
- Reports include raw expression/R2, fitted expression/R2, structure, endpoint action metrics,
  basin occupancy, source-target pairing cost, and runtime/candidate counts.
- Relative Fisher loss is reported against the exact zero-velocity predictor globally and by
  readout/op/arg/inactive block kind; absolute `1e0` loss is not an acceptance condition.
- Learned-flow eval reports every deterministic `theta0` draw plus task-level medians, quantiles,
  failure-mode frequencies, oracle-free sample consensus, and explicitly labeled best-of-N metrics.
- A longer medium run is permitted only after a short diagnostic shows that high-leverage readout
  probabilities/margins improve and the result is not explained only by coefficient fitting.
- The active v4 mainline has exactly one readout, uses learned rollout after bootstrap, and records
  zero mutation/elite/archive participation.
- No global semantic posterior, block-marginal target, Sinkhorn plan, or Hungarian assignment is
  created in active v4 training; the corresponding legacy CLI controls are not launch parameters.
- The simplified active update uses no KL posterior or cross-source OT.  Every accepted target keeps
  its original source lineage and minimizes raw semantic loss inside a bounded local Fisher cell
  neighborhood.
- Semantic proximal search is not attempted until learned reference endpoints pass the terminal
  single-expression manifold-gap gate; a failed gate is reported before expensive expression scoring.
- Learned rollout evaluation reports literal GT expression hits/mass/cell distance separately from
  fitted R2, medoid, or semantic-energy indicators.
- Eval retraction always preserves the decoded expression, restores inactive source identity, and
  reports both all-block and active-only mean/p95 Fisher displacement rather than hiding it.

## Errors

| Error | Attempts | Resolution |
|---|---:|---|
| Strict endpoint baseline stopped at epoch 16 with no output files | 1 | Treat as interrupted evidence; do not claim validation |
| `acos` Fisher distance returned `4.8e-7` at identical distributions | 1 | Use the equivalent stable half-angle `atan2` formula |
| GPU smoke and direct CUDA preflight both stalled before first output | 2 | Stop all processes; use CPU pipeline validation and report GPU environment blocker separately |
| First CPU smoke produced NaN proposer post-update metrics | 1 | Stabilize square roots in FR loss and keep proposal/evaluation on the same soft endpoint |
| Stratified cycle-time patch landed in archived score trainer | 1 | Restore score sampling and place five-bin rotation in `train_cycle_flow` |
| Whole-function cleanup patch missed exact trailing context | 2 | Switched to generated context-only `apply_patch` hunks; clean `run()` replacement succeeded |
| Planning patch first targeted the outer workspace cwd | 1 | Retried with absolute paths inside `SFSR/SemanticFlowSR` |
| Medium GPU run failed entering cycle collection summary due CPU/GPU tensor mix | 1 | Move semantic elite weights to the benchmark device before coupling and summary reductions |
| CPU reference-proposer smoke used removed `--nguyen-limit` CLI flag | 1 | Use `--suites nguyen` plus existing task limits |
| First reference-proposer medium left `gt_traces_per_task=1`, collapsing reference argmax diversity | 1 | Stop the run and rerun medium with `gt_traces_per_task=4` and lower projection sharpness |
| Retraction unit fixture `x0+x0` simplified to unsupported constant `2*x0` | 1 | Use representable `x0+x0**2`; no algorithm change required |
| First v3 CPU smoke rejected its only task at correction budget | 1 | Keep the 0.25 gate; persist a rejected cycle and report the best attainable ratio instead of crashing |
| 2-task v3 GPU short diagnostic stalled in trace semantic collection | 1 | Stop after bootstrap evidence; deduplicate/canonicalize traces before expensive semantic scoring and cache expression metrics |
| Legacy checkpoint policy test expected the old flag spelling | 1 | Update assertion to the new `--legacy-cycle-eval` alias; behavior was correct |
| GT endpoint had nonzero distance to its own cell | 1 | Make lineage cell projection idempotent when selected active probabilities already exceed `1-epsilon` |
| DSO smoke failed with `ModuleNotFoundError: semflow_sr` in `dso37` | 1 | External envs do not have the repo installed; rerun baseline restoration with `PYTHONPATH=$PWD` so manifest runners can import local package code. |
| Baseline command-plan redirect failed because `results/benchmark_plans/` did not exist | 1 | Create the directory before redirecting matrix script stdout. |
