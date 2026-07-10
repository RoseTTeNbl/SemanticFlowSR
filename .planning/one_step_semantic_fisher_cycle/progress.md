# Progress

## 2026-07-10

- Read repository instructions and inherited Stage1/semantic endpoint planning records.
- Confirmed no training process remains active.
- Inspected the interrupted strict register endpoint log through epoch 16.
- Mapped the main script's compiler, endpoint predictor, Fisher geometry, semantic endpoint,
  evaluation, and output functions.
- Created this scoped plan for the one-step semantic Fisher cycle.
- Added `semflow_sr/one_step_fisher.py` with semantic log-quality weights,
  systematic resampling, source-preserving Hungarian coupling, and FR map loss.
- Added five focused geometry/coupling tests. The first run found an `acos`
  identity-distance precision issue; the implementation now uses stable half-angle geometry.
- Focused geometry and register-flow tests pass (`11 passed`).
- GPU 1 was idle, but both the tiny training command and a direct CUDA preflight stalled
  before output. Both were terminated and process cleanup was verified.
- Fixed semantic-credit alignment so expression samples score the same soft endpoint
  particle that is later resampled and transported.
- Removed inactive endpoint filling from the new cycle; all legal blocks travel together.
- Second CPU smoke completed with finite diagnostics and a small decrease in proposer FR loss.
- Replaced matrix-style temporal HTML with register execution trajectories and snapshots.
- Replaced the algorithm/architecture mainline docs and added the new math derivation and failure record.
- Re-ran focused tests after the five-bin time-sampling correction: `12 passed`.
- Audited legacy references and enumerated obsolete result/log directories for deletion.
- Restricted construction dispatch to `register_categorical_blocks`, removed latent imports,
  latent training/evaluation, and old runtime dispatch; the trainer still compiles.
- Removed semantic endpoint correction buffer/training functions and simplified bootstrap to
  the Fisher bridge path.
- Resumed to implement the flow-first semantic tilt plan: v-rollout proposals, trace-level
  graph elites, source-conditioned sharp projection, active-only semi-discrete Fisher coupling,
  and graph simplex visualization.
- Added stable Fisher-Rao probability path/logit velocity helpers plus source-conditioned
  sharp trace projection and active-only trace atom coupling. Focused one-step tests pass:
  `tests/test_one_step_fisher.py` -> `9 passed`.
- Replaced cycle collection with flow-rollout proposals, trace-level expression elites,
  tracked-source semi-discrete coupling, optional one-step student gating, new CLI defaults,
  and graph simplex output hooks. Static compile passes; help shows the new cycle flags.
- CPU smoke after fixing candidate evaluation produced valid trace-level candidates
  (`valid_expression_sample_rate=1.0`), improved selection score from about `1.19` to
  `0.69`, kept `tracked_theta0_reuse_rate=1.0`, and wrote `graph_simplex_flow.html`.
- Two-iteration CPU run stayed finite with valid candidates both rounds, ESS about
  `1.6-1.7`, unique elite active graph count `3`, and t-bin flow diagnostics populated.
- Eval smoke verified student gate failure skips proposer eval and flow eval still runs.
- Updated README plus algorithm, math, and architecture docs to describe v2 flow-first tilt.
- Started medium GPU benchmark request. The helper shell script only supports smoke/overfit,
  so this run will call `scripts/train_complete_expression_semantic_fm.py` directly with
  medium parameters and three outer iterations. GPU 0 has the most free memory but is busy;
  GPUs 1-3 are memory-saturated by other processes.
- First medium GPU attempt completed both proposer and flow bootstrap, then failed at
  cycle collection summary because elite weights were CPU tensors and selected score tensors
  were CUDA tensors. Patched collection to use device-local semantic weights.
- Parsed the medium retry logs: both runs stopped before outer-cycle selection. The bootstrap
  evidence is still actionable: `G_phi` active argmax match topped out near `0.46` then fell
  to `0.43`, while `v_psi` ended with norm ratio around `0.66-0.70`; neither should be the
  sole endpoint authority while diagnosing the semantic tilt.
- Added `cycle_proposer_source=gt_reference` as a diagnostic proposer that constructs endpoint
  probabilities directly from the nearest strict compiled trace for the tracked `theta0`, and
  added `cycle_projection_sharpness` to keep active targets entropy-controlled instead of
  nearly one-hot. Focused tests pass: `29 passed`.
- A tiny CPU smoke with `--cycle-proposer-source gt_reference --cycle-projection-sharpness 0.7`
  ran through collection and flow training. It produced valid expression samples, improved
  selection score from about `0.257` to `0.067`, and kept raw/fitted tilted R2 at about
  `0.881/0.959`.
- Started a medium GPU run with `gt_reference` and sharpness `0.7`, but stopped it after
  cycle 1 because `gt_traces_per_task` was left at the default `1`, making
  `unique_argmax_endpoint_count=1.0` and invalidating the multi-endpoint check.
- Re-ran medium GPU with `gt_traces_per_task=4`, `projection_sharpness=0.6`, no one-step
  student, and three outer iterations. The run completed and wrote
  `results/clean_benchmark_20260701/ablations/complete_expression_semantic_fm_20260707/runs/gt_reference_tilt_medium_gpu3iter_gt4sharp06_20260710_codex`.
  Cycle summaries were stable: tilted raw/fitted R2 ended around `0.931/0.976`, ESS around
  `3.79`, top weight around `0.28`, pair FR cost around `1.47`, and flow loss around `0.91`.
  Final flow eval remained poor because rollout endpoints stayed high-entropy and underpowered:
  raw R2 mean `-9.19`, fitted R2 mean `0.428`, terminal entropy `0.789`, terminal max prob
  `0.354`, active argmax match `0.45`.
- Fixed future output routing so `cycle_flow_diagnostics.jsonl` contains cycle-flow t-bin
  diagnostics instead of duplicating graph visualization rows. Static compile and focused
  one-step tests pass after the patch (`10 passed` for `tests/test_one_step_fisher.py`).
- Added prior score median/p90/max summary fields because medium iterations 2-3 had huge
  score means driven by a few pathological candidates; static compile and focused tests
  still pass (`10 passed`).
- Added a `ReferenceFisherBridgeOracle` eval path behind `--eval-reference-field-oracle`.
  It generates endpoints with compiled trace targets and the analytic Fisher bridge closed-form
  ODE solution, using neither `G_phi` nor learned `v_psi`. Eval-only from the medium checkpoint
  shows the oracle solves the three compiled Nguyen eval tasks exactly (raw/fitted R2 and
  skeleton all `1.0`) while learned flow eval remains structurally collapsed. This confirms
  the current failure is flow realization/generalization rather than the reference endpoint.
- Replaced the active outer-loop proposer with `cycle_proposer_source=reference_bridge`.
  The cycle now samples endpoint atoms directly from the previous iteration's selected
  coupling/reference bridge archive; iteration 1 falls back to compiled-trace seeds only.
- Removed one-step student/proposer instantiation, training, eval, checkpoint state, and
  sample-output mixing from the active cycle. `--cycle-one-step-student` now defaults false
  and raises if enabled; `--cycle-proposer-source flow_rollout` is rejected.
- Added expression-level graph-fiber collapse before simplex projection. Elite rows now record
  `expression_collapse_key`, duplicate counts/excess, and representative choice keys; collapse
  tie-breaks prefer current-round candidates over archive replay so iteration-2 coupling
  diagnostics correctly show `previous_elite_archive`.
- Verified with static compile and regression tests:
  `/home/ywj/miniconda3/envs/semflow/bin/python -m py_compile scripts/train_complete_expression_semantic_fm.py`
  and `tests/test_one_step_fisher.py tests/test_trace_compile.py tests/test_semantic_mass_ng.py`
  passed (`29 passed`).
- CPU two-iteration smoke `reference_bridge_cycle_min_smoke2_20260710_codex` confirmed
  `reference_bridge_seed_rate` transitions `[1.0, 0.0]`; proposals, elites, and couplings
  report `{1: compiled_trace_seed, 2: previous_elite_archive}`.
- Eval smoke `reference_bridge_cycle_eval_smoke2_20260710_codex` confirmed normal sample
  outputs contain only `model_role=fisher_velocity_field`; `reference_field_eval` is `not_run`
  and `reference_field_samples.jsonl` is empty without the explicit oracle flag.
- Started a full reference-bridge medium run on GPU0 with the intended 3-iteration settings,
  but interrupted it after misreading a long output gap as collection slowdown. The run had
  already entered cycle-flow training; this is treated as an operator interruption, not an
  algorithm failure.
- Re-ran the full medium configuration on GPU1 as
  `reference_bridge_medium_gpu1_3iter_20260710_codex`. It completed 3 outer iterations plus
  normal learned-flow eval. Key cycle quantities: seed rate `[1.0, 0.0, 0.0]`, unique endpoint
  argmax `[1.0, 4.0, 4.0]`, tilted raw/fitted R2 ending at `0.779/0.945`, ESS ending at
  `3.753`, and flow loss/norm ratio improving to `0.586/0.759`.
- Checked output JSONL files. Proposals, elites, and couplings all report
  `{1: compiled_trace_seed, 2: previous_elite_archive, 3: previous_elite_archive}`. Normal
  samples contain only `model_role=fisher_velocity_field`; reference-field oracle rows are
  absent.
- Inspected final sample expressions. They are valid but structurally wrong or too simple:
  `(x0 + x1)**2`, `sqrt(Abs(x0**3))`, `x0`, and `x0**3`. Fitted R2 can be high, but raw R2
  and skeleton remain poor. Wrote the full run analysis to
  `results/clean_benchmark_20260701/ablations/complete_expression_semantic_fm_20260707/runs/reference_bridge_medium_gpu1_3iter_20260710_codex/analysis_summary.md`.
- Implemented hard-endpoint population + multi-term readout changes: additive register compiler,
  term-level fit diagnostics, hard endpoint eval, disabled soft endpoint sampling by default,
  per-iteration eval files, expression-level graph-fiber collapse, and GPU script output-terms=3.
  Focused/regression tests passed: `35 passed`.
- Interrupted medium run `hard_endpoint_multiterm_medium_gpu3_3iter_20260710_codex` was stopped
  after enough evidence was collected. Iteration 1 showed healthy target-side exploration
  (candidate unique expressions `14.86`, multi-term rate `0.42`, tilted raw/term-fit R2
  `0.93/0.99`) but eval hard endpoints remained under-sharp and often single-term
  (terminal max prob `0.38`, term-count mean `1.0`, skeleton accuracy `0`).
- Fixed eval theta0 coverage: deterministic per-task/per-index random starts now produce distinct
  `theta0_hash`/argmax keys and each sample row stores the full `eval_theta0_population`.
- Added cycle-flow block type weights to emphasize readout/op blocks (`--cycle-readout-loss-weight`,
  `--cycle-op-loss-weight`) and selected-side term-count diagnostics.
- CPU smoke `hard_endpoint_population_eval_cpu_smoke_20260710_codex` completed two iterations.
  It confirmed `theta0_unique_hash_count=4`, population unique expression count `4`, selected
  multi-term rate `1.0`, and archive source transition to `previous_elite_archive` in iteration 2.
- Error: accidentally ran `git diff -- ... | --stat`, which failed with `--stat: command not found`.
  Re-ran the correct command as `git diff --stat -- ...`.
- Cleaned result directories under
  `results/clean_benchmark_20260701/ablations/complete_expression_semantic_fm_20260707/runs`.
  Removed temporary CPU smokes, duplicate smoke/eval runs, and partial hard-endpoint medium runs.
  Retained only:
  `reference_bridge_medium_gpu1_3iter_20260710_codex`,
  `gt_reference_tilt_medium_gpu3iter_gt4sharp06_20260710_codex`, and
  `gt_reference_tilt_medium_gpu3iter_gt4sharp06_reference_eval_20260710_codex`.
- Wrote handoff for the next agent:
  `.planning/one_step_semantic_fisher_cycle/HANDOFF_20260710.md`.
  It documents the active algorithm, retained evidence, cleaned directories, current failure mode,
  and recommended next diagnostics.
- Resumed from the handoff with no training process running and results already cleaned.
- Read the repository constraints, complete planning skill instructions, active/legacy planning
  records, handoff, and dirty-worktree inventory. Switched the active plan from completed history
  cleanup back to `one_step_semantic_fisher_cycle`.
- Added gated continuation phases: first localize failure from retained evidence, then add relative
  and per-block diagnostics, run a bounded diagnostic, make only evidence-backed algorithm changes,
  and finally broaden population evaluation.
- User approved implementation of the simpler v3 constrained single-expression plan after rejecting
  a more expensive semantic-Wasserstein variant.
- Restored the activity plan, confirmed no training process is running, and recorded v3 phases and
  invariants before touching algorithm code.
- Re-read the active plan and audited the dirty worktree plus the current Fisher helper/tests.
  Corrected a stale v2 acceptance clause that incorrectly required previous-elite archive replay.
- Bumped the objective identifier to v3 and added finite-particle KL-proximal weighting, generic
  log-domain Sinkhorn transport, and source-conditioned entropic trace coupling with automatic
  correction-penalty search/posterior shrinkage. Legacy Hungarian helpers remain untouched.
- Added focused property tests for the v3 probability layer; `tests/test_one_step_fisher.py` passes
  with `13 passed` in the `semflow` environment.
- Canonicalized commutative register expressions before deterministic SSA emission, exposed CSE/SSA
  compiler diagnostics, enforced one readout at the v3 training entry point, and added exact semantic
  tests for a cubic polynomial, mixed add/sub, and a shared subexpression.
- The compiler/Fisher focused set now passes with `22 passed`; the single-readout construction phase
  is complete and the active work moves to replacing the heuristic outer-loop collector.
- Audited the full v2 collection/training path and chose an isolated v3 collector rather than adding
  more conditionals to mutation/elite/archive code.  Legacy functions will remain non-active.
- Mapped the learned RK2 rollout, legal categorical sampler, raw register semantic execution, and
  current affine-fitted energy.  Defined the v3 empirical law as direct temperature-1 complete traces
  plus a decaying training-only GT mixture.
- Imported the v3 KL/Sinkhorn primitives into the trainer and extended coupled examples with soft-plan
  mass, complete trace identity, and a training-only GT-anchor flag.
- Added hard register-trajectory execution, signed-pair reachability scoring, raw v3 trace semantics,
  canonical expression-fiber aggregation, decaying GT-mixture mass support, and the specified four-term
  semantic energy. The trainer still compiles after this insertion.
- Added the isolated v3 collector: learned RK2 reference rollout, direct complete-trace sampling,
  GT-anchor mixture, KL-proximal posterior, full soft Sinkhorn edge materialization, correction-budget
  rejection/shrink diagnostics, and zero mutation/elite/archive counters. Static compilation passes.
- Switched the active outer loop to the v3 collector, added `cycle_expression_posterior.jsonl`, left
  the legacy elite file empty, updated algorithm/coupling summary labels, and changed CLI defaults to
  64 direct traces, KL budget 0.10, GT-alpha auto schedule, OT ratio 0.25, entropy scale 0.05, and
  zero mutation/elite/archive participation.
- Static compile and CLI help pass after the runner switch. Began the semantic-conditioner phase by
  auditing the model's local feature construction and confirming the two existing register semantic
  switches were previously redundant.
- Split register conditioning into soft-bank and hard-prefix views, added two signed-pair reachability
  features per action, widened the register semantic feature width to 10, and enabled both views in
  the v3 direct-velocity model.
- Static compilation plus the compiler/Fisher/semantic regression set passes after semantic widening
  (`38 passed`). Mapped the remaining v3 objective work: weighted soft-plan sampling, register GT
  direction teacher, separate inactive drift, terminal RK2 consistency, and blockwise zero baselines.
- Added reusable per-block Fisher loss values, the low-time register GT semantic-direction teacher,
  and an 8-step-compatible differentiable RK2 terminal rollout helper in preparation for rewriting
  the cycle optimizer.
- Rewrote `train_cycle_flow` for weighted soft-plan sampling, 25% GT replay, separate
  `L_FM + 0.05 L_inactive + 0.10 L_GT`, a `0.25 L_endpoint` RK2 pass, global/readout/op/arg
  relative-to-zero diagnostics, low-time relative bins, and inactive drift reporting. Static compile
  passes.
- Audited summary/checkpoint/CLI serialization after the objective rewrite; the new loss and model
  architecture fields still need to be persisted before checkpoint compatibility tests.
- Persisted v3 objective weights and soft/hard conditioner metadata, added all terminal/teacher CLI
  defaults, and implemented explicit read-only v2 evaluation compatibility. V2 cannot resume or start
  v3 training.
- Added no-GT terminal single-expression retraction and integrated it into normal learned-flow eval.
  Each population row now contains pre/post expression, pre/post sharpness, active/inactive FR jump,
  expression-preservation, and task-population mean/p95 retraction cost.
- Added v3 hard-decode/retraction guards, summary fields, CLI controls, and a unit test covering
  idempotence, expression preservation, selected-action sharpness, and exact inactive-fiber identity.
- First retraction test run reached `38 passed` plus one fixture failure because the parser simplified
  `x0+x0` into an unsupported constant multiplier. Replaced only the fixture with `x0+x0**2`.
- Retraction/semantic/Fisher focused tests pass with `39 passed`. Replaced coefficient-fit candidate
  selection with an oracle-free raw semantic population medoid and relabeled best-of-N R2 as GT oracle
  diagnostics.
- Added v3-specific tests for expression-fiber aggregation/I-projection (no hybrid block
  recombination), finite soft+hard register-conditioned direct velocity, and an inference method
  signature containing only `(X, y, theta, t, theta0)` with no GT/teacher/trace argument.
- Focused v3 plus legacy compiler/Fisher/semantic regression suite passes with `45 passed`.
- Updated the GPU helper to the v3 contract: one readout, learned rollout, KL/GT/OT controls, zero
  mutation/elite/archive, inactive/teacher/terminal loss defaults, and 16/64 trace smoke/overfit budgets.
- Ran the first end-to-end CPU smoke. Bootstrap was finite, then the sole task was correctly rejected
  by the correction gate and the run stopped because rejection was still raised as an exception.
  Next change is to persist that outcome and expose its best attainable correction ratio.
- Added `CorrectionBudgetError(best_ratio, limit)`, persisted rejected-task summaries, and changed the
  runner to skip the outer optimizer (without changing the field) when every task is rejected.
- Core tests after rejection handling pass (`26 passed`). CPU smoke
  `v3_cpu_pipeline_smoke2_20260710_codex` completed, wrote v3 artifacts/checkpoint, skipped the invalid
  outer update, and ran oracle-free medoid + terminal retraction evaluation.
- Added and passed a direct v3 optimizer test covering soft-plan weighting, forced GT replay,
  semantic-direction teacher, inactive drift, per-kind relative baselines, and differentiable terminal
  consistency (`tests/test_semantic_fisher_v3.py`: `4 passed`).
- Replaced the active `train_stage1` bootstrap call with source-conditioned GT atom examples trained by
  `train_cycle_flow` under the full v3 objective. The old stage1 trainer remains only as legacy code.
- Persisted bootstrap example/relative-loss summaries; static compile and v3 Fisher tests pass after
  the bootstrap switch (`17 passed`).
- CPU bootstrap smoke `v3_cpu_bootstrap_smoke_20260710_codex` completed with finite FM, inactive,
  teacher, terminal, relative, and per-kind diagnostics, then correctly skipped the still-infeasible
  outer correction.
- Corrected replay diagnostics to distinguish the actual fraction of GT-anchor examples from the
  forced 25% replay branch, and added oracle-free medoid versus explicitly GT-oracle best-of-N fields
  to task-level eval summaries.
- Added a checkpoint-policy test ensuring v2 loads only under explicit read-only legacy eval. Core
  collection and semantic/objective/retraction implementation phases are now complete; regression and
  bounded diagnostic validation remain.
- Full focused v3/legacy regression suite passes with `47 passed`. No SemanticFlowSR training process
  is active. GPUs 0-1 are saturated; GPU3 has about 6.5 GB free and is suitable only for a tiny bounded
  diagnostic.
- Started bounded GPU3 run `v3_gpu_short_diag_20260710_codex`. Stopped it during collection after
  several minutes with no new output; bootstrap metrics were already far below the required gates.
  Process cleanup is confirmed, and no medium/8-task run was started.
- Moved expensive v3 semantic scoring after expression-fiber aggregation and reduced the per-layer
  signature shortlist from 8 to 2 while retaining exhaustive raw signed-pair NMSE evaluation.
- Read the new lineage-preserving proximal theory in full and pivoted the active implementation plan:
  remove KL/Sinkhorn from the main update, gate on terminal manifold realization, solve a small local
  per-source proximal problem, add literal GT-generation diagnostics, and add the requested 3-row
  first/middle/final ODE coupling visualization. No further training run is authorized by current evidence.
- Bumped the objective to v4 lineage-proximal and added the new collector implementation: fixed-source
  learned rollout, own-cell manifold gap, literal hard/sample/trace-mass GT probes, small active-block
  cell-neighbor enumeration, raw semantic MAP inside a Fisher radius, and same-source target pairs.
- Switched the active runner/output summary to the lineage collector, removed KL/Sinkhorn objects from
  active outputs, added low-mass training-only GT replay, and made global expression sampling a rejected
  legacy option (`--cycle-expression-samples 0`). Added radius/budget/manifold/GT-probe CLI controls.
- Added literal GT-hit and fail-fast manifold-gate tests. Fixed lineage cell projection to be idempotent
  for endpoints already inside their decoded epsilon cell; lineage-focused tests now pass (`7 passed`).
- Added fixed-source RK2 snapshots and the requested first/middle/final outer-iteration landscape.
  Parameter panels use a common Hellinger/Fisher PCA; expression panels use a common raw-semantic PCA.
  Learned trajectories, accepted local targets, and diagnostic GT cells have distinct line/marker styles.
- The active collector now writes no global expression posterior: failure is rejected at the own-cell
  manifold gate before semantic scoring; a passing lineage evaluates at most six expensive local semantic
  candidates and keeps the same `theta0` without Sinkhorn/Hungarian recoupling.
- Simplified the GPU helper to `--cycle-expression-samples 0` and local radius/candidate/gate controls;
  removed KL/GT-mixture/correction/OT arguments from the active launch command.
- Added an explicit Chinese failure/cost/GT-generation analysis in
  `V4_LINEAGE_PROXIMAL_ANALYSIS_20260711.md`. Static compilation and the focused compiler/Fisher/semantic/
  lineage suite pass with `51 passed`; no new GPU experiment was started.
- The first tiny v4 CPU integration exposed inactive-block dilution in the manifold metric: full-block
  mean/p95 appeared to pass even though the only active decoded block was far from its cell. Changed all
  lineage cell/gate distances to active blocks only and added a regression test for this invariant.
- Repeated the same one-task/one-source CPU integration after the fix. It correctly rejected before any
  semantic candidate execution: active own-cell mean/p95 `2.533/2.533`, hard/sample GT hit `0/0`, GT trace
  geometric mean `0.151`, nearest GT-cell FR RMS `1.988`, and candidate count `0`. SVG/PNG/JSONL landscape
  artifacts were still written, so failure remains visible rather than blocking diagnostics.
- Added the same direct GT probes before terminal retraction in ordinary eval plus active-only retraction
  mean/p95. A read-only CPU eval confirmed hard/sample GT hit `0/0`; active retraction mean/p95 were
  `1.645/1.898` while the diluted all-block mean was only `0.433`. Focused tests now pass with `52 passed`.
- New user-directed phase started: clear `results/`, launch a detached v4 medium train/eval, diagnose
  it while running, and delete inactive historical interfaces from the trainer. The running process
  will be launched before source cleanup; its exact command/source boundary and PID will be recorded.
- Preflight: `results/` is `289M`; no SemanticFlowSR trainer is active. GPU 3 has about `24.1 GB` free
  and is selected. Historical medium budget was recovered as 12/4 tasks, hidden 128, particles 8,
  three outer iterations, bootstrap 6 epochs, and outer flow 6 epochs x 60 steps.
- Added an explicit `SCALE=medium`: 12 train / 8 eval tasks, hidden 128, particles 8, bootstrap 6x60,
  outer flow 3 iterations of 6x60, and 8 deterministic eval theta0 draws. Shell syntax passes.
- Pre-launch source boundary: git `bd915795e9e4b1ef8ed5ed20473c5019cfb0d3c7`; trainer SHA256
  `60eece771372a051b6595ceb68231fd5fd6c8c59dcd88d5a824a0db45ba984a0`; runner SHA256
  `0aeb09b7150fd4160c6923a2bbac8fc3bdf380fec784d255d9198d7b236c4205`.
- Cleared generated contents under `results/` and preserved only tracked `results/README.md` (`8K`).
- Error: interpreted the cleanup request too broadly and removed baseline artifacts that the user
  intended to retain. Code cleanup is paused; recovery from same-host copies/archives/rebuild sources
  is now the active priority. No additional deletion is authorized.
- Recovery search found no usable duplicate in isolated workspaces, OneDrive, Trash, paper workspace,
  git history, or open deleted handles. Host filesystem is ext4 without an exposed snapshot and this
  session has no passwordless sudo. Proceeding to reconstruct the baseline tree from source outputs,
  logs, manifests, and paper assembly scripts.
- User clarified the required recovery method: rerun all repository external-baseline scripts in the
  provided environments and rebuild the complete organized baseline result tree/data. This supersedes
  filesystem undelete attempts; source cleanup remains paused.
- Detached medium run launched on physical GPU 3. Wrapper PID `3310329`, Python PID `3310370`, tag
  `v4_lineage_medium_gpu3_3iter_20260711_codex`; startup log and exact CLI were verified. Initial GPU
  allocation was ~258 MiB while data/compiler setup began.
