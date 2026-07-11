# Findings

## 2026-07-10 Baseline audit

- The strict register endpoint baseline stopped after epoch 16 and wrote no checkpoint or evaluation output.
- Its best endpoint categorical NLL was about `0.503`; active argmax accuracy stayed near `0.58`, so it did not establish endpoint recovery.
- Existing code already has strict trace compilation, register action masks, analytic Fisher bridges,
  endpoint semantic evaluation, and coefficient fitting before semantic energy.
- Existing semantic endpoint correction reuses one model and does not implement an explicit
  source-preserving empirical coupling or a proposer-update phase.
- The new route requires two trainable objects with separate checkpoints and metrics:
  one-step endpoint proposer `G_phi` and legal tangent field `v_psi`.

## Design decisions

- Use systematic resampling to realize the tilted endpoint marginal with a fixed particle count.
- Use minimum-cost one-to-one assignment from fresh source particles to the resampled endpoint
  particles, with Fisher distance as cost. This preserves the empirical source marginal exactly.
- Train `v_psi` only on analytic Fisher bridge states/tangents from those paired particles.
- Distill `G_phi` on exactly the same source-endpoint pairs; do not independently shuffle targets.
- Keep raw and coefficient-fitted semantic metrics side by side.
- Semantic quality samples and transported endpoint particles must share the same
  soft endpoint distribution. Crediting sampled expressions to an unrelated argmax
  active mask would recreate the earlier false-target problem.
- Two-cycle validation confirms the update loop is real, but fitted-only semantic
  tilt can worsen raw numerical behavior: unique endpoints fell from 4 to 2 and
  unfitted raw MSE rose from about 1.95 to 11.18 while fitted distance improved.
- The next cycle must make `v_psi` the reference proposer. The existing independent
  `G_phi` proposer and fresh source resampling break the interpretation of the outer
  iteration because the velocity learned in one round is not what produces the next
  round's endpoints.
- Sharp endpoint targets should be conditioned on the assigned source particle:
  active trace blocks become epsilon-smoothed one-hot choices, while inactive blocks
  preserve the source probabilities. This keeps the target atomic in the decoded graph
  quotient without forcing irrelevant simplex coordinates to move.
- Medium bootstrap logs did not actually reach semantic cycle metrics. They show an
  endpoint-generation problem before the cycle: the optional one-step student/proposer
  only reached about `0.43` active argmax accuracy, and the direct flow bootstrap only
  reached about `0.68` target norm ratio. Using either as the endpoint authority can
  under-move particles and produce overly averaged endpoint distributions.
- The v2 CPU flow-rollout cycle had high source-target pair FR cost (`~4.0-4.5`) and
  near-zero cycle-flow norm ratio in the tiny setting, consistent with targets that were
  too sharp/far for the learned flow to recover. A teacher/reference proposer is needed
  as a diagnostic path to isolate tilt/coupling from learned-rollout error.
- Medium GPU with `gt_reference`, `gt_traces_per_task=4`, and projection sharpness `0.6`
  shows the semantic target construction is no longer the primary failure: tilted raw/fitted
  R2 reaches about `0.93/0.98`, ESS stays about `3.8` over four elite modes, and active graph
  diversity stays at four modes. The remaining failure is endpoint realization by the learned
  direct velocity field: cycle-flow norm ratio only improves from about `0.35` to `0.41`, and
  final flow eval endpoints remain high entropy (`~0.79`) with low max probability (`~0.35`).
- Graph-fiber duplication is material: medium elites had 28 unique active graph keys per
  iteration but only 11-16 unique expressions. Several tasks use multiple construction graph
  fibers for the same decoded expression, which spreads probability mass in the simplex even
  when expression-space selection is correct.
- `one_step_cycle_samples.jsonl` is final evaluation output from the learned flow rollout,
  not the cycle target/elites. High tilted R2 in `cycle_history` measures selected training
  targets before model rollout; it does not imply the learned flow endpoint realizes those
  targets during eval.
- A reference-field oracle eval that uses compiled traces plus the analytic source-conditioned
  Fisher bridge reaches exact raw/skeleton results on the three compiled Nguyen eval tasks.
  The same checkpoint's learned flow still emits simple/collapsed expressions with high
  terminal entropy and low active-target probabilities. This isolates the failure to learned
  flow realization/generalization, not the reference endpoint construction.
- The active outer-loop proposer has been changed from learned rollout/student endpoints to
  a direct reference-bridge endpoint sampler. Iteration 1 seeds from compiled traces only
  because no archive exists; iteration 2+ samples from the previous iteration's selected
  expression atoms and then redoes expression-level exploration/tilt/coupling.
- Expression-level graph-fiber collapse must be applied before projecting back to the simplex.
  Otherwise multiple construction graph fibers for the same decoded expression absorb separate
  elite slots and diffuse target mass. The active selector now collapses by decoded expression,
  records duplicate counts/choice keys, and uses one representative graph fiber for projection.
- Normal eval remains learned-flow inference only. The reference-field oracle is still available
  behind `--eval-reference-field-oracle`, but its rows are written separately to
  `reference_field_samples.jsonl` and are not mixed into `one_step_cycle_samples.jsonl`.
- Reference-bridge medium GPU validation confirms the new outer loop is logically closed:
  iteration 1 uses compiled trace seeds, while iterations 2-3 use `previous_elite_archive`
  in proposals, elites, and couplings. The archive sampler increases endpoint mode diversity
  (`unique_argmax_endpoint_count` from 1 to 4) and improves flow matching alignment
  (loss `0.787 -> 0.653 -> 0.586`, norm ratio `0.653 -> 0.723 -> 0.759`).
- The target side is now mostly healthy: selected elites remain valid, expression-level unique
  modes stay at four per task, and exact/near-exact expressions appear in the archive. Graph
  fiber duplication is still visible in candidates, but expression collapse reduces selected
  duplicate excess to about `0.57-1.43` per task.
- Final learned-flow eval still fails structurally despite better training alignment. Samples
  are valid expressions, but they are simple or fitted-only approximations such as `(x0+x1)**2`,
  `sqrt(Abs(x0**3))`, `x0`, and `x0**3`; skeleton and operator-dependency accuracy remain
  zero. Raw R2 median is only about `0.108`, while fitted R2 median is about `0.913`, showing
  coefficient-fit masking rather than true symbolic recovery.
- Hard-endpoint multi-term rerun evidence separates target-side and flow-side failures. In the
  interrupted medium run, iteration-1 target construction was healthy enough to explore:
  candidate unique expressions averaged `14.86` per task, candidate multi-term rate was `0.42`,
  tilted raw/term-fit R2 was about `0.93/0.99`, and cycle flow loss fell to `0.94` with norm
  ratio `0.70`. Eval hard endpoints still had terminal max probability only `0.38`, skeleton
  accuracy `0`, and term-count mean `1.0`. Therefore the main issue is not absence of expression
  candidates but weak realization of selected multi-term readout targets by the learned flow.
- Eval previously reported only the best candidate over several `theta0` starts. That hid the
  actual pushforward population. The active eval now uses deterministic per-task/per-index random
  `theta0` samples, records `theta0_hash`/argmax keys, and stores the full per-`theta0` hard
  endpoint population in each sample row.
- Register readout blocks are structurally high-leverage: three readout choices determine whether
  multi-term expressions survive hard decoding, but they previously received the same loss weight
  as ordinary argument blocks. Cycle flow now upweights readout and operator blocks by default
  (`readout=3.0`, `op=1.5`) while keeping inactive identity handling unchanged.

## 2026-07-10 learned-endpoint diagnostic handoff

- No algorithm code has been changed in this continuation yet. The retained evidence first needs
  to distinguish objective normalization, block imbalance, time-local integration error, and
  insufficient terminal categorical margin.
- Target construction is not the leading hypothesis: the `gt_reference` target population reaches
  roughly `0.93/0.98` tilted raw/fitted R2, and the analytic reference-field oracle exactly recovers
  the compiled tasks. Learned inference alone collapses to simple expressions with skeleton and
  operator-dependency accuracy `0`.
- The main discontinuity is continuous probability transport followed by hard graph decoding.
  A moderate aggregate tangent error can cross an argmax boundary in one readout/op block and
  replace the whole expression; therefore endpoint action margin and blockwise relative error are
  more causal diagnostics than aggregate Fisher loss or fitted R2.
- Current absolute loss is ambiguous without its exact zero-predictor denominator. Averaging over
  legal blocks can conceal a large relative error in only three readout blocks, and inactive/source-
  identity blocks may make an aggregate look better without improving the decoded active graph.
- Multi-sample inference is allowed as an oracle-free inference aid, but it must not be reported only
  as best-of-N against GT. Useful population measurements include per-task expression mode frequency,
  pairwise semantic agreement on observed `X`, stability across `theta0`, consensus/medoid quality,
  and separately labeled oracle evaluation of every sample and best-of-N scaling.
- Process inspection confirms no SemanticFlowSR training is active. GPU 1 is currently fully free
  (about 24 GB available); it is reserved only for a short diagnostic after metric instrumentation.
- Earlier structural-closure evidence found severe low-time error hidden by aggregate loss. The
  current cycle uses stratified time sampling, but the retained summaries do not establish whether
  relative error and readout/op error improve in every time bin. This must be measured directly.
- Retained result listing contains the three handoff medium runs plus a current CPU two-iteration
  artifact and a `_legacy_failed_20260710` quarantine directory. No result cleanup is needed before
  the next bounded diagnostic.
- The generic CLI default is `endpoint_bridge`, but the active one-step cycle constructor explicitly
  hardcodes `velocity_parameterization="direct_velocity"`. Therefore current multi-term cycle runs
  train and numerically integrate `v(theta_t,D,t,theta0)`; the generic CLI value is irrelevant to this
  mainline. This corrects an earlier provisional inference made before reading the constructor.
- Current algorithm/architecture prose still contains legacy `G_phi`, one-readout, and generic ODE
  descriptions that do not fully match the handoff's removed-student, three-readout active path.
  Documentation should be corrected only after the executable diagnosis is settled.
- Current multi-term logs confirm bootstrap and outer cycle both use Fisher probability-velocity
  regression under `direct_velocity`; there is no bootstrap/outer categorical-objective switch in
  the active path. The endpoint-map-loss hypothesis applies only to a dormant generic parameterization
  and is not a proposed mainline fix.
- The retained medium's low-time loss remains high after three rounds (`~0.68-0.72`) while the easy
  `t>0.1` bin is `~0.27`. Aggregate `0.586` is therefore optimistic even before normalizing by the
  zero predictor.
- Only eight tracked sources per task are coupled to four modes. Re-training on a fresh 8-point
  Hungarian assignment each round asks a high-dimensional `(D, theta0)->atom` network to extrapolate
  a semi-discrete transport partition from very sparse labels. This is a second structural cause of
  new-`theta0` collapse and should be measured via train-source versus fresh-source endpoint accuracy.
- Checkpoint inspection shows the retained `reference_bridge_medium_gpu1_3iter` used
  `template.output_terms=1`; the three-readout implementation was added afterward. That artifact is
  valid evidence for the endpoint-realization bottleneck but cannot validate the current multi-term
  readout construction or its new weights.
- Current multi-term active-graph traversal deliberately marks every readout slot active, including
  slots whose target is ZERO. Thus unused readout suppression is supervised; arbitrary activation of
  an untrained readout slot is not the primary construction bug.
- The source-conditioned target probabilities have decisive selected-action margin, but the direct
  field must realize them through the full low-time trajectory. Recent multi-term logs retain losses
  around `0.9-1.3` in low-time bins while `t>0.1` is near `0.37`; endpoint failure is consistent with
  unresolved early-trajectory error rather than a closed-form endpoint classifier.
- A concrete target/training mismatch remains: inactive target blocks equal `theta0`, but both the
  current runner and bootstrap set their loss weight to zero. Shared network updates can still move
  these blocks during rollout, and the model consumes the full evolving state. Per-inactive drift and
  an identity-weight ablation are therefore required diagnostics.

## 2026-07-10 v3 implementation decisions

- User approved the simpler v3 plan: direct velocity remains authoritative; mainline output has one
  readout; mutation, elite selection, and previous-elite archive leave the active path.
- Semantic correction is a KL-budgeted exponential tilt over complete trace particles. This is the
  minimum practical implementation of the semantic proximal/I-projection chain and avoids the later,
  rejected high-dimensional semantic-Wasserstein design.
- Source correction uses soft log-domain Sinkhorn with exact empirical marginals. The cost combines
  source-to-sharp-target Fisher distance and reference-rollout-to-target Fisher distance; posterior
  strength is shrunk when the correction ratio cannot meet `0.25`.
- Existing register semantic functions are reusable. The v3 field will receive both soft-state and
  hard-prefix semantic features; GT remains training-only through anchor/replay and a low-time
  semantic teacher loss.
- The dirty worktree contains extensive historical changes. Implementation should add isolated v3
  helpers/tests and avoid reverting, formatting, or committing unrelated paths.
- The active planning acceptance list still contained a v2 archive-replay requirement. It has been
  replaced by the v3 learned-rollout/no-archive invariant before implementation begins.
- `semflow_sr/one_step_fisher.py` is an untracked v2 helper module, so v3 probability primitives can
  be added there without modifying legacy callers; the old Hungarian/capacity functions will remain
  available only for explicit legacy paths.
- The v3 coupling is naturally a law over `(source, complete trace)` rather than a source-independent
  endpoint table: `P_epsilon(z; p0)` depends on the row source.  The implementation therefore keeps
  exact uniform source and trace-posterior marginals in a Sinkhorn plan and materializes a sharp
  endpoint for every nonzero source/trace edge.
- The correction budget may be infeasible for a fixed semantic column marginal, because changing
  only the assignment cannot move its mass back toward the reference law.  The helper first increases
  `lambda_corr`; only if that fails does it shrink posterior strength toward the pre-proximal trace
  prior, and it rejects the update if even the prior violates the budget.
- Focused v3 probability tests validate the actual invariants: the KL update lowers expected energy
  without exceeding its budget, Sinkhorn matches both marginals, and source-conditioned endpoints
  preserve inactive blocks while meeting the correction ratio.
- The register compiler already emitted bottom-up SSA and cached exact expression keys.  Making the
  expression tree canonical before emission is sufficient to turn that cache into deterministic CSE;
  no new graph representation is needed.  Multi-readout additive flattening remains reachable only
  for legacy templates, while the v3 runner rejects any `output_terms != 1`.
- The current collector is tightly bound to reference/archive atoms, mutation-generated candidates,
  top-mode selection, capacity resampling, and hard Hungarian assignment.  Reusing it conditionally
  would leave too many hidden v2 branches, so v3 should call a separate collector and emit its own
  complete-posterior diagnostics.
- Existing categorical sampling already respects every block's legal action mask.  For v3, direct
  temperature-1 samples from the learned rollout define the empirical complete-trace prior; graph
  fibers are aggregated by complete decoded expression before proximal weighting, with representative
  choices retained only to materialize a sharp atom.
- The current semantic energy performs a global affine fit before its signature score, which is
  incompatible with v3.  The new collector must compute raw NMSE/signature directly from the decoded
  output and leave all affine/term-fit quantities as diagnostics only.
- Exact signed-pair reachability contains `O(R^2)` candidates per layer.  A practical implementation
  can batch all raw NMSE terms and evaluate the existing richer signature only on a small NMSE
  shortlist; this retains the graph-aligned `h_r`, `h_a+h_b`, `h_a-h_b` geometry without making
  64-trace diagnostics prohibitively expensive.
- `CycleCoupledExample` previously represented one hard Hungarian assignment and had no probability
  mass.  V3 examples now need a `sample_weight` carrying the soft plan edge, the complete target trace
  choices, and a GT-anchor marker so training can sample the full coupling without capacity slots.
- Canonical recompilation provides a deterministic representative after expression-level fiber

## 2026-07-11 external baseline restoration

- The active baseline restoration path is the repository-owned paper-complete workflow:
  `scripts/run_paper_complete_benchmarks.sh` followed by
  `scripts/build_paper_complete_results.py`.
- The expected source-run buckets are
  `results/clean_benchmark_20260701/external_baselines/formula_dev/` and
  `results/clean_benchmark_20260701/external_baselines/symbolicgpt_large/`.
- The expected paper-readable outputs are under
  `results/clean_benchmark_20260701/paper_complete_20260702/` with `source_runs/`,
  `final_tables/`, `per_task/`, `manifests/`, `logs/`, and `trained_small_models/`.
- The current SFSR repo has no surviving baseline result JSON after the user's results cleanup;
  `logs/` only retains SemanticFlowSR training logs. Historical search under `/home/ywj/wyh/results`
  and `_internal/experiments/sfsr_isolated_20260702` did not find copyable SFSR baseline outputs.
- Required conda environments are present and startable: `semflow`, `deap`, `pysr`, `dso37`, and
  `tpsr`. The official TPSR checkpoint and local diffusion proposal files are present.
- `scripts/run_paper_complete_benchmarks.sh` has been upgraded to cover the full external-baseline
  method surface used by `configs/eval/external_baselines.yaml`: GP, GP-DEAP, PySR, DSO, TPSR, E2E,
  LocalDiffusionProposal, SymGPT-small, NeSymReS-small, HVAE-small, and NGGP-small. Its smoke checks
  confirm every method can write the shared baseline JSON schema and the paper-complete builder can
  ingest the resulting files.
  aggregation.  The empirical prior mass still sums every sampled graph fiber; only endpoint
  materialization uses the canonical SSA representative, so duplicate fibers cannot become separate
  posterior modes.
- V3 energy components are computed only from training observations: raw output NMSE, raw semantic
  signature distance, hard-register signed-pair reachability, and expression complexity.  Existing
  affine/term-fit values remain attached to particle rows solely as diagnostics.
- The soft coupling can be represented without stochastic resampling in memory: one example is kept
  per numerically nonzero Sinkhorn edge and sampled later according to its edge mass.  This is an
  unbiased finite representation of the plan and keeps exact source/target marginal diagnostics.
- A correction-budget failure is now a first-class collection outcome.  Rejected tasks produce
  posterior diagnostics but no training pairs; if every task is rejected, training stops before an
  invalid bridge objective is formed.
- The active runner now has executable v3 guards rather than descriptive flags: any nonzero mutation,
  elite count, archive size, soft-endpoint sampling, non-learned proposer, multi-readout chart, or
  non-atomic projection sharpness is rejected before training.
- The model currently routes both semantic feature switches to the same soft register computation.
  V3 needs two distinct views: soft-bank candidate consequences from the current probability state,
  and a hard-argmax prefix bank that exposes discrete structural reachability.  The existing local
  action head can accept these by widening register semantic features from 8 to 10 dimensions.
- Signed-pair conditioning is kept inference-safe: its two extra per-action values are the best
  normalized NMSE reachable by one register `+/-` composition and the gain over the action alone.
  They use only the current state, `X`, and observed `y`; hard-prefix projection is internal and does
  not expose GT traces or symbolic labels.
- The legacy semantic teacher helper is fixed-symbol-specific and cannot supervise register op/readout
  blocks correctly.  V3 should derive a register teacher distribution directly from the hard-prefix
  reachability feature, then bias the known GT active action during training-only replay.
- Inactive blocks have a zero target tangent, so the literal zero-predictor relative ratio is undefined
  (`0/0`).  Diagnostics should report their absolute Fisher drift and normalize it by the global active
  zero-predictor loss, while readout/op/arg use ordinary per-kind relative-to-zero ratios.
- The GT teacher is formulated as a training-only tangent-direction target, not an extra inference
  input: hard-prefix signed-pair reachability defines a smooth action distribution, the known GT action
  receives a finite bias on active blocks, and the predicted Fisher tangent is aligned by cosine only
  for low-time GT replay examples.
- Terminal consistency is implemented as a bounded auxiliary optimization pass after each FM epoch.
  Keeping it in small RK2 minibatches avoids retaining the computation graph for all 16 rollouts at
  once while preserving the requested 8-step differentiable endpoint supervision.
- The checkpoint metadata did not previously record semantic feature switches, so a state dict alone
  could be reconstructed with the wrong local-head width.  V3 must persist both soft/hard conditioner
  flags and reject incompatible objective versions before model construction.
- Legacy compatibility is intentionally read-only.  A v2 objective is accepted only when both
  `--eval-only` and `--legacy-v2-eval` are present; it reconstructs the old semantic-feature width from
  checkpoint metadata/defaults and writes outputs that retain the v2 objective label.
- Eval retraction uses the learned endpoint itself as the inactive-fiber base, matching
  `P_epsilon(decode(theta_tilde); p_tilde_1)`.  It preserves inactive probabilities exactly and
  short-circuits when active selected probabilities already exceed `1-epsilon`, which makes the
  executable map idempotent despite the raw epsilon-mixture formula not being algebraically idempotent.
- Because v3 retraction is defined by a single hard decode, soft-sample endpoint decoding would create
  a second incompatible trace law after projection.  The active v3 runner therefore requires hard
  argmax decode; multi-sample evaluation comes from independent deterministic `theta0` draws.
- The previous eval selector still optimized a coefficient-fit-aware score.  V3 now selects the raw
  semantic population medoid (pairwise signature distance, target-energy/complexity only as tie-breaks).
  Test R2 maxima remain available under explicitly named `eval_gt_oracle_best_of_n_*` diagnostics.
- The first minimal CPU smoke had only two GT-bootstrap optimizer steps; its direct field norm ratio
  was about `0.0013`.  Every sharp endpoint correction therefore violated the `0.25` trust ratio.
  This validates the rejection gate but also shows that a rejected outer update must be serializable
  rather than terminating the whole run.
- Correction failure now carries the best ratio attained at maximum penalty/prior shrinkage.  This
  distinguishes a genuine trust-region infeasibility from a generic Sinkhorn error and lets summaries
  retain the quantitative gate failure.
- The second CPU smoke completed the entire rejected-update/eval/checkpoint path.  Its KL proximal
  step exactly used the `0.10` budget, lowered expected energy `0.549 -> 0.397`, and lowered weighted
  raw NMSE `1.109 -> 0.758`; all mutation/elite/archive counters were zero.
- The same smoke correctly failed both downstream realization gates: correction best ratio was about
  `1.28e5` because reference source-to-endpoint cost was only `7.8e-6`, and eval retraction FR
  mean/p95 were `0.409/0.512` despite 100% expression preservation.  These values must not be treated
  as a successful diagnostic run.
- The active GT bootstrap was still using the old FM-only trainer.  V3 now materializes
  source-conditioned compiled GT atoms and applies the same inactive, semantic teacher, and terminal
  consistency objective before the first learned rollout; this removes a bootstrap/outer objective
  mismatch without exposing GT at inference.
- CLI bootstrap smoke confirms every v3 loss is active and finite.  With only two optimizer samples,
  global/readout/op/arg relative losses all remained about `1.0`, terminal loss was `4.40`, and the
  outer correction was still rejected.  This is execution evidence, not convergence evidence.
- The bounded 2-task GPU diagnostic completed two bootstrap epochs but failed the continuation gates:
  global relative Fisher loss improved only `0.998 -> 0.989`; readout/op/arg ended near
  `0.989/0.986/0.991`; terminal consistency remained `5.16`; norm ratio was only `0.019`.
- Trace semantic collection then took multiple minutes even for only `2 tasks x 4 sources x 8 traces`.
  The current implementation canonicalizes and computes full hard-register reachability separately
  for every raw sample, including duplicate expressions.  Expensive semantic metrics must move after
  cheap expression/fiber aggregation and be cached by canonical expression key before any 8x8x64 run.
- The optimized collector now performs only canonical identity/decode work per raw sample, aggregates
  empirical mass, and then scores each unique expression once.  All signed-pair raw NMSE candidates
  remain batched and exhaustive; rich signature distance is evaluated for the best two NMSE candidates
  per layer instead of eight.

## 2026-07-11 lineage-proximal simplification

- The new theory identifies the correct update as endpoint support motion along each existing source
  lineage, not global reweighting and recoupling.  The active chain should be
  `theta0 -> learned bar_theta -> local semantic/Fisher proximal theta+ -> same-source FM`.
- Smoke evidence says `bar_theta` is not yet a meaningful expression endpoint: relative Fisher loss
  is about `0.989`, velocity norm ratio about `0.019`, terminal consistency about `5.16`, and the CPU
  source-to-reference Fisher cost is near zero.  Hard-decoded/sample expressions are therefore prior
  artifacts; KL energy improvement cannot be interpreted as GT generation.
- A global Sinkhorn plan cannot repair this regime.  The huge correction ratio comes from asking an
  almost stationary flow endpoint to jump to an epsilon-sharp cell, not from poor assignment.
- The cheap, falsifiable algorithm is: first gate on distance to the endpoint's own decoded cell; only
  then enumerate a small deterministic set of nearby complete-expression cells, choose the minimum
  raw semantic loss inside a Fisher radius, and retain the same `theta0`.
- Final semantic energy should use only decoded raw output behavior (raw NMSE/signature); intermediate
  register semantics may rank which local cell neighbors to evaluate, but should not add another
  expensive per-layer reward.
- Direct evidence must count whether learned rollout argmax or a small fixed number of samples equals
  GT, plus GT-trace probability and nearest-GT-cell distance.  Fitted R2 and posterior energy are only
  secondary diagnostics.
- Expensive semantic work is now fail-fast.  Before the learned endpoint reaches its own decoded cell,
  the active collector performs zero local semantic evaluations.  After it passes, structural proposal
  enumeration may inspect active-block alternatives cheaply, but raw semantic execution is capped at
  six complete expressions per source by default instead of 16/64 global samples.
- The paper visualization must use a single embedding basis across all selected outer iterations and
  ODE times.  Parameter vectors are `2*sqrt(p)` so Euclidean geometry matches the Hellinger/Fisher chart;
  expression vectors are raw output semantic signatures.  GT cells are diagnostics, not flow points or
  selection targets, and are drawn separately from accepted proximal targets.
- Manifold and retraction gates must be active-block statistics.  Averaging unchanged inactive blocks
  reduced the first v4 CPU own-cell mean from the true active value `2.533` to `0.133`, which would have
  incorrectly admitted an almost-zero flow.  Eval now keeps the old all-block number for compatibility
  but reports active mean/p95 as the causal endpoint-realization metric.

## 2026-07-10 cleanup audit

- Focused geometry/register tests pass after the stratified-time fix (`12 passed`).
- The active trainer still imports and dispatches the discarded latent-component model and
  the old semantic endpoint/pushforward correction flows.
- Historical complete-expression artifacts currently occupy seven run directories and five
  top-level logs; none is the final stratified validation.
- The clean mainline should retain only `one_step_semantic_fisher_cycle`; archived token
  scripts and root planning chronology are historical material, not active interfaces.
- After removing public legacy dispatch, the remaining complexity is dead online semantic
  guidance/projection code and a score network still embedded in the trainer; the active cycle
  always evaluates with guidance off and does not need those objects.

## 2026-07-11 medium relaunch and cleanup

- Before the requested cleanup, `results/` occupied about `289M` and contained paper/baseline archives
  plus v1-v3 SemanticFlowSR runs. The user explicitly requested clearing the results directory, so no
  result artifact is being treated as retained state; quantitative history already lives in this plan.
- GPU 3 is the only fully idle device (`~24.1 GB` free, 0% utilization). GPUs 0/2 are saturated and
  GPU 1 is active, so the v4 medium run should be pinned to GPU 3.
- Historical medium sizing was consistently `12 train tasks / 4 eval tasks`, hidden `128`, 12 layers,
  8 source particles, 3 outer iterations, 6 bootstrap epochs, and 6x60 outer flow updates. The new
  launch should retain that training budget while increasing eval coverage rather than reviving the
  old global-expression sampling/OT path.
- `results/README.md` is the only tracked file under `results/`; cleanup must preserve it and delete
  generated top-level result trees only. There are no result-directory symlinks.
- Repository policy confirms the only active entrypoints are the complete-expression trainer and the
  one-step GPU wrapper. Target-conditioned reference, latent endpoint, semantic endpoint correction,
  and token pushforward are explicitly historical and are safe candidates for interface removal from
  the active trainer, provided unrelated modules/worktree edits are preserved.
- Static audit: the active trainer is `7,798` lines and `one_step_fisher.py` is `932` lines. Two wholly
  unreferenced collectors occupy roughly lines `4,872-5,658`: mutation/archive/Hungarian v2 and
  KL/Sinkhorn v3. The old graph-simplex HTML writer and the earlier stage1 trainer are also uncalled.
- The CLI still publicly exposes latent, syntax-prior, reference-attractor, score-network, online
  guidance, semantic-endpoint, KL/OT, mutation/archive, soft-decode, proposer, and legacy-checkpoint
  flags even though v4 either ignores or rejects them. This is the largest readability/interface
  problem and can be removed independently of the running process.
- `one_step_fisher.py` still exports old resampling, Hungarian, KL and Sinkhorn dataclasses/functions;
  current tests preserve those historical APIs. Cleanup should first determine the exact v4 imports,
  then rewrite tests around the retained Fisher path/projection primitives before deleting helpers.
- Baseline recovery audit after the mistaken deletion: same-host isolated-result directories are empty;
  no matching copy exists under OneDrive, the paper workspace, other `results/` roots, Trash, or open
  deleted file handles. Git history contains only a much older edge-flow smoke, not the July paper
  baseline tree. `/home` is ext4 with no exposed snapshot and noninteractive sudo is unavailable, so
  direct filesystem undelete is not currently viable from this session.
- The remaining recovery path is reconstruction from benchmark manifests, baseline source outputs,
  root logs, and the paper-result assembly scripts. No further cleanup/refactor should proceed until
  the recoverable baseline scope is enumerated and rebuilt.
