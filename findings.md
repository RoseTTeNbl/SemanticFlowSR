# Findings

## 2026-07-09 Iterative Corrected-Bridge Stage2 Update

- Latest theory update changes the operational Stage2 interpretation to iterative endpoint-corrected flow matching:
  `v^k -> theta1_ref -> terminal semantic tilt -> theta1_plus -> new bridge (theta0, theta1_plus) -> v^(k+1)`.
- The existing graph collection path already rolled out the current field once, sampled complete expressions from `theta1`, applied semantic terminal tilt, and projected to `theta1_plus`. The critical mismatch was the default base freezing / `xy_residual` route, which made the practical default a residual ablation instead of training the next full corrected field.
- Graph Stage2 mainline now uses objective version `iterative_semantic_endpoint_corrected_bridge_fm_v1`, cache signature fields `training_mode=corrected_bridge_fm`, `train_full_field=true`, and bridge pair `theta0_to_semantic_tilt_projected_theta1_plus`.
- `--semantic-endpoint-training-mode corrected_bridge_fm` is the default. `residual_ablation` and explicit `--no-semantic-endpoint-train-base` are retained only for legacy comparison.
- Tiny graph Stage2 train-only smoke `graph_stage2_corrected_bridge_smoke_trainonly_20260709` verified the code path:
  - `semantic_endpoint_training_mode=corrected_bridge_fm`;
  - `semantic_endpoint_train_full_field=true`;
  - trainable params equal total params (`259202`), proving the base field was not frozen;
  - endpoint collection wrote the new cache and correction summary.
- Tiny eval smoke `graph_stage2_corrected_bridge_smoke_eval_20260709` verified sample output and endpoint probability diagnostics:
  - 2 Nguyen tasks, `r2_mean=0.9865`, raw R2 mean `0.9564`;
  - `endpoint_trace_family_best_active_mean_prob=0.9476`, argmax match `1.0`;
  - sample rows include GT, raw expression, affine expression, R2, raw R2, and active endpoint probabilities.
- This smoke is an interface and plumbing validation only. It reuses a tiny Stage1 smoke checkpoint and does not prove full-scale Stage2 quality.
- Full-data Stage1 8-epoch training observed earlier took about 47 minutes from launch to epoch 8 completion, with best loss `0.00818`; the prior full eval was slow because it had no partial writes before the recent eval-progress patch.
- Tiny graph Stage2 collection timing was about 16.3 seconds for 2 endpoints with `ode_steps=4`, `semantic_endpoint_samples=2`, and 1 projection-check sample, or roughly 8 seconds per endpoint under intentionally small settings. Full settings scale primarily with `buffer_size * ode_steps * (semantic_endpoint_samples + projection_check_samples)`.

## 2026-07-09 Full Graph Stage1 Run Monitoring

- Full graph Stage1 run `graph_target_conditioned_stage1_full_20260709_r1` is active under `scripts/run_graph_target_conditioned_stage1_gpu.sh` on `cuda:0`.
- Output target: `results/clean_benchmark_20260701/ablations/complete_expression_semantic_fm_20260707/runs/graph_target_conditioned_stage1_full_20260709_r1`.
- Log target: `logs/complete_expression_semantic_fm/graph_target_conditioned_stage1_full_20260709_r1.log`.
- First epoch reached training loss `0.08991084497247356` versus zero-pred loss `0.5989418806415051`, with cosine `0.972767845261842` and norm ratio `1.0011066338307864`. This is early evidence that the target-conditioned reference flow is fitting the Stage1 velocity target on full data; final judgment still depends on eval R2 and samples.
- By epoch 3, full Stage1 loss had decreased to `0.022395529280474877`, cosine increased to `0.9936886429786682`, and norm ratio stayed near target at `0.9750550415739417`. This strongly supports convergence of the Stage1 reference-flow regression on the full configuration.
- By epoch 4, loss decreased further to `0.017905442236806266`, cosine `0.994816890321672`, and norm ratio `0.9782904290035367`.
- Per-sample endpoint probability fields are expected in `typed_op_node_flow_samples.jsonl`, including active block probability/logprob and trace-family-best endpoint diagnostics, once eval completes.
- After the user clarified that 8 epochs are sufficient for the first full-data check, the 24-epoch run was stopped before final output writing and replaced by `graph_target_conditioned_stage1_full_e8_20260709_r1`, still using the full dataset/full graph config but with `--epochs 8`.
- The 8-epoch full-data run completed training with stable convergence: epoch losses `0.0899 -> 0.0404 -> 0.0224 -> 0.0179 -> 0.0101 -> 0.00818 -> 0.00894 -> 0.00954`; best loss was `0.008180854120000731` at epoch 6. Epoch 8 cosine remained high at `0.9943187788501382` and norm ratio was `1.007710670530796`.
- After epoch 8, the process remained alive with CPU/GPU activity but had not yet written final output files, consistent with full eval/rollout running before `write_outputs()`.

## 2026-07-09 Read-Only Diagnostic Setup

- The workspace has substantial pre-existing uncommitted changes, deleted files, and untracked result scripts/planning files; this diagnostic will not revert or clean them.
- Top-level records indicate many endpoint semantic-pushforward revisions (v3-v11) mostly tightened diagnostics, but previous medium runs still failed held-out R2/structure despite positive semantic-mass statistics.
- Active `.planning/fixed_symbol_node_stage1` records a cleaner fixed-symbol-node Stage1/Stage2 path that previously normalized to high sample-level metrics; this must be compared against the current token and graph result artifacts rather than assuming the current mainline is healthy.
- Current algorithm docs define the intended object as endpoint-only semantic pushforward: `theta1 -> q_theta1(z|D) -> semantic KL tilt by complete-expression target distance -> q_plus(z|D) -> theta1_plus -> supervised FM correction`. This explicitly makes endpoint semantic statistics necessary but not sufficient for final SR success.
- Token docs require the Stage2 mainline to preserve weighted complete traces (`weighted_trace_fm`) because per-position marginals can recombine individually good tokens into syntactically or semantically bad expressions.
- Graph docs require a Stage1 reference gate before interpreting Stage2: low loss, high cosine/norm ratio, high rollout terminal max probability, and valid expression production.
- Current code and gate config use `SEMANTIC_ENDPOINT_OBJECTIVE_VERSION = endpoint_semantic_signature_pushforward_projection` without the v11 suffix described in earlier progress records. That means result/cache interpretation must check actual summaries and timestamps, not rely on the historical v11 narrative.
- Current result tree has only these retained complete-expression semantic FM sample-producing runs: token Stage1 eval, token Stage2 eval, token Stage1 sample32 eval, and graph Stage1 train-only. The graph run has `typed_op_node_flow_samples.jsonl` with 0 rows and `n_tasks=0`.
- Token Stage2 eval (`token_semantic_pushforward_weighted_trace_medium_eval_20260709`) has 207 sample rows, but sample-level quality is poor despite positive endpoint statistics: `r2_mean=0.108836`, solution rate `0.028986`, skeleton accuracy `0.057971`, op-dependency accuracy `0.019324`, valid expression fraction `0.789050`.
- Token Stage2 worsens sample structure relative to token Stage1 eval: constant-only predictions rise from 35/207 (`16.9%`) to 53/207 (`25.6%`), `R2 >= 0.95` falls from `8.7%` to `4.8%`, and paired task comparison has 116 worse vs 89 improved with median R2 delta `-0.0157`.
- Token Stage2 endpoint training summary reports strong internal semantic-posterior numbers (`tilt_accept_rate=0.919922`, target-distance improvement `3.578773`, projected target-distance improvement `3.584958`), so the failure is not visible from endpoint aggregate metrics alone.
- Existing token medium collection checkpoint was created in an older schema without `cache_signature` or `semantic_endpoint_objective_version`; current code has signature checking, but this existing result should be treated as old-format evidence, not a clean current-objective run.
- Graph Stage1 medium run stops at train-only evidence in the current filesystem. Its Stage1 gate is clearly failed (`stage1_best_loss=1.199245`, cosine `0.910621`, norm ratio `0.791716`, low-t losses around `2.57-3.21`) and the expected eval output directory mentioned in the log is absent.
- The `.planning/fixed_symbol_node_stage1` high-score normalized result path is not present in the current result tree, so it cannot be used as directly reproducible current evidence without restoring or rerunning that artifact.
- Token sample-level R2 is heavily inflated by affine refit relative to the raw generated expression. In token Stage2 eval, median raw expression R2 is about `-14.04` while median affine-refit R2 is only `0.0063`; 110/207 rows have raw R2 below `-10`.
- Read-only checkpoint diagnostic on 24 eval tasks shows the token Stage2 terminal distribution is sharp but mostly off-grammar: terminal max-prob mean is `0.834`, but mean probability mass on grammar-allowed tokens along the decoded prefix is only `0.364` and the mean per-task minimum allowed mass is `0.113`. Decode-time grammar masking is therefore changing/renormalizing the learned distribution rather than sampling from a grammar-respecting flow.
- Token code confirms why the terminal confidence is misleading: `encode_formula()` returns all-ones weights over all 64 token positions, including EOS/PAD tail positions, and `weighted_trace_fm` also trains every sampled trace endpoint with all-ones position weights. The reported active block count of 64 means most loss/cosine can be dominated by padding/end-of-sequence structure rather than the semantic expression prefix.
- Token Stage2 target inspection: sampled 296 `theta1_plus` endpoint examples from the 8192-example collection checkpoint. Only `88.2%` decoded within a 0.75s timeout; 30 hit parse-budget limits, 1 timed out, and 4 decode errors occurred. Among decoded examples the median R2 was about `-0.0116`, `64.4%` had negative R2, and the sample weighted R2 mean was strongly negative due to pathological outliers.
- The sampled Stage2 endpoint-teacher traces are structurally weak: class counts were 111 constant-only, 72 single-variable affine/scaled, 77 compound, 30 budget-skip, 4 decode-error, 1 timeout, 1 single-simple-term. This means the semantic endpoint buffer itself contains many bad complete traces before the correction model ever trains on it.
- High posterior-weight trace examples often match only affine-normalized sampled semantics, not GT structure. Examples include GT `log(x0 + 1.4) + log(x0**2 + 1.3)` with endpoint trace `x0-5**1.78/1.35` at weight `0.5435`, and GT `x1**3` with a long affine-ish `x1` expression at weight `0.5390`.
- Code-level mismatch: token Stage1 supervises all 64 positions including PAD (`encode_formula`, `endpoint_weight = ones`), and token Stage2 `weighted_trace_fm` also sets `weight=torch.ones((seq_len,))` for every sampled endpoint trace. This makes PAD/tail and arbitrary sampled-token positions part of the velocity target, not just active semantic structure.
- Code-level mismatch: token Stage2 `weighted_trace_fm` samples complete traces from the current model endpoint `theta1` and turns every sampled sequence into a sharp endpoint teacher, weighted by semantic posterior. If the model's support is mostly constants, affine surrogates, or complex numerical expressions, the correction trains toward those self-sampled artifacts.
- Code-level mismatch: the mainline tilt uses `--semantic-tilt-energy target_distance`, and `semantic_tilt_energy(..., mode="target_distance")` drops complexity/collapse penalties from the posterior weights. The energy diagnostics include collapse/complexity penalties, but the actual posterior weight can still prefer collapsed or affine-surrogate traces if their normalized semantic signature distance is small.
- Code-level mismatch: both semantic energy and eval use affine/output normalization. `expression_semantic_energy` compares standardized output/signature, and eval selects candidates by train-set affine-refit R2 minus a tiny complexity penalty. This explains high R2 for wrong structures and posterior preference for expressions like scaled `x0`, constants plus `x`, or exponent approximations.

## 2026-07-09 Continued Read-Only Diagnosis

- Recomputed current sample statistics using `/home/ywj/miniconda3/envs/semflow/bin/python`; the default shell `python` is not available, so repo commands should use the `semflow` conda interpreter or activate that environment first.
- Current docs define the intended runtime contract as:
  - activate/use `semflow` Python;
  - run from `SemanticFlowSR/`;
  - graph entrypoint `scripts/train_complete_expression_semantic_fm.py`;
  - token entrypoint `scripts/train_token_policy_semantic_fm.py`;
  - Stage2 only after Stage1 gate.
- The current code constant is `SEMANTIC_ENDPOINT_OBJECTIVE_VERSION = "endpoint_semantic_signature_pushforward_projection"` in `scripts/train_token_policy_semantic_fm.py`, matching the gate docs, not the older recorded v11 suffix narrative.
- Token Stage2 eval summary remains misleading if read without samples:
  - `n_tasks=207`;
  - `r2_mean=0.108836`, `r2_median=0.006302`;
  - `solution_rate=0.028986`;
  - `skeleton_accuracy=0.057971`;
  - `operator_dependency_accuracy=0.019324`;
  - `valid_expression_fraction_mean=0.789050`;
  - `rollout_terminal_max_prob_mean=0.833870`;
  - semantic endpoint aggregate fields are positive: tilt accept `0.919922`, target-distance improvement `3.578773`, projected target-distance improvement `3.584958`.
- Reclassified token eval samples by variable usage rather than literal numeric form:
  - Token Stage1 eval has 35/207 no-variable constant expressions (`16.9%`).
  - Token Stage2 eval has 53/207 no-variable constant expressions (`25.6%`).
  - Token Stage2 has only 10/207 exact single-variable identity predictions (`4.8%`) and 24/207 multi-variable predictions (`11.6%`).
- Token Stage2 Nguyen examples show complete semantic/grammar drift:
  - Nguyen-3 GT `x0**5 + x0**4 + x0**3 + x0**2 + x0`, prediction `4.0`, skeleton `C`, R2 `-0.003786`, raw R2 `-3.780484`.
  - Nguyen-5 GT `sin(x0**2)*cos(x0) - 1`, prediction `1.4142135623731*cos(1.35)+4-(2.7-8+2.0+x0)`, R2 `-0.043459`, raw R2 `-2884.820557`.
  - Nguyen-6 GT `sin(x0) + sin(x0 + x0**2)`, prediction `5`, skeleton `C`, R2 `-0.002084`, raw R2 `-20.561964`.
- Token Stage2 versus Stage1 paired comparison across 207 common tasks:
  - 116 tasks worsened and 89 improved by affine-refit R2;
  - median R2 delta `-0.015742`;
  - 9 solved Stage1 tasks became unsolved under Stage2, while 4 became solved.
- Token Stage2 raw expression fit is much worse than affine-refit metrics:
  - median raw R2 without affine refit is `-14.042396`;
  - 110/207 rows have raw R2 below `-10`;
  - high affine R2 rows can still be structurally wrong, e.g. `x2*0.426/1.7**Abs(5)` for GT `x2` has affine R2 `1.0` but raw R2 `0.058972` and op-dependency mismatch.
- Stage2 train curve confirms all 64 token positions are supervised in the retained medium run: `active_block_count_mean=64.0` for each epoch. This is consistent with `encode_formula()` and `weighted_trace_fm` setting all token position weights to one, including EOS/PAD tail.
- Code-level token distribution mismatch:
  - `rollout()` integrates unconstrained per-position logits over the full vocabulary, then `integrate()` recenters positions; no grammar mask is applied during ODE integration.
  - `sample_sequence()` and `argmax_sequence()` apply `grammar_allowed_mask()` only at final decode time by zeroing/masking forbidden tokens and renormalizing.
  - Therefore terminal sharpness over the full vocabulary does not imply that the learned flow puts probability mass on grammar-legal prefix continuations.
- Code-level posterior/projection mismatch:
  - token `semantic_endpoint_group_diagnostics()` computes projected diagnostics for `weighted_trace_fm` by multinomially resampling from the same collected posterior support, not by rolling out the trained correction model and decoding from its terminal distribution.
  - Positive projected target-distance diagnostics therefore prove the sampled endpoint buffer can be reweighted, not that the trained token policy will generate those expressions.
- Existing token weighted-trace collection checkpoint is old schema evidence:
  - `semantic_endpoint_collection_checkpoint.pt` has top-level keys `construction_family`, `examples`, `semantic_endpoint_group_count`, `semantic_endpoint_token_projection`, `semantic_tilt_energy`, `target_count`;
  - it lacks `cache_signature` and objective-version metadata;
  - use it as failure evidence for the retained run, not as clean current-objective validation.
- Graph branch has no current sample-level result to inspect:
  - only `graph_stage1_syntax_endpoint_dual_gate_medium_20260709/typed_op_node_flow_samples.jsonl` exists and it has zero rows;
  - no separate graph Stage1 eval output directory exists under the retained result tree;
  - no real graph/token training process is currently running.
- Graph retained Stage1 train-only summary fails the documented gate:
  - `stage1_best_loss=1.199245` versus gate `<0.30`;
  - cosine `0.910621` versus gate `>0.98`;
  - norm ratio `0.791716` versus gate `>0.90`;
  - low-t losses remain high (`3.106972`, `3.207004`, `2.573163`);
  - `n_tasks=0`, so R2/valid-expression summaries are missing-eval artifacts, not evidence of graph SR behavior.
- Current graph runner logic is conceptually correct: it trains Stage1, evaluates Stage1, checks the gate, and skips graph pushforward if the gate fails. The retained graph artifact stopped at train/eval-start state, so the immediate problem is to recover/rerun Stage1 eval evidence, not to interpret graph Stage2.

## Prior Findings
- Full Stage1 bridge-path training reached low FM loss but failed rollout sharpness and structure recovery.
- Exact teacher rollout reaches endpoint, so the integration and endpoint bridge are not fundamentally broken.
- Model alignment is poor near `t=0`, causing rollout to leave the reference tube.
- Low-t continuation did not fix the failure, suggesting direct path velocity is under-conditioned near random initial states.
- Endpoint-attractor diagnostics are being tested as a cleaner reference field without auxiliary losses.

## 2026-07-08 Endpoint-Attractor Diagnostic
- Run: `stage1_attractor_diag_symgpt200_h256_e6_20260708`.
- GPU run completed successfully.
- Final Stage1 loss improved to `1.081`, with cosine `0.969` and norm ratio `1.019`.
- Low-t bins remain high: `<0.001` loss `6.38`, `0.001-0.005` loss `6.16`, `0.005-0.02` loss `4.26`.
- Rollout is sharper than bridge-path (`terminal_max_prob_mean=0.518`) but endpoint/structure remains weak (`endpoint_active_prob_mean=0.102`, skeleton/opdep `0`).
- Conclusion: Stage1 does not pass the gate; do not run formal Stage2 full validation yet.

## 2026-07-08 Semantic Feature Diagnostics
- Fixed `copy` semantic execution: it now returns its input instead of being silently converted to zero.
- Vectorized fixed-symbol action consequence feature construction; feature shape/finite smoke passed on GPU.
- 100-step fixed-batch direct endpoint-attractor + semantic features reduced loss from zero-pred baseline `6.56` to `4.11`, but needs longer training for a real overfit conclusion.
- 100-step endpoint-bridge + semantic features was worse (`loss=5.29`, cosine `0.45`), so endpoint-bridge is not the current Stage1 direction.
- `gt_traces_per_task=4` with random copy assignment was stopped before training because it spent minutes in trace compilation with only CPU active; full use needs a trace cache or a bounded compiler path.
- `bridge_plus_random` initially had a sampler fallback bug; fixed.
- Off-path random-state endpoint-attractor with `min_remaining=0.05` is theoretically inconsistent and produced huge targets (`loss ~= 116`) because random high-t states were divided by `1/(1-t)`.
- New random-state diagnostic uses `endpoint_attractor_min_remaining=1.0`, i.e. constant-strength local Fisher attraction on off-path states.
- Constant-strength random-state diagnostic did not work as a Stage1 replacement: after 4 epochs, loss was still `5.32`, cosine `0.40`, and low/mid t bins stayed high. It was stopped before slow eval finished.
- Current Stage1 candidate remains bridge-path endpoint-attractor; random off-path states should not be used as a high-probability replacement without a more structured curriculum.
- Added `task_encoder_mode=stats|hybrid_stats`, using only inference-available `X_train,y_train` global statistics and basis correlations. This targets the `t≈0` failure where `theta_t` carries almost no endpoint information.
- `hybrid_stats` GPU smoke passed; no GT trace or intermediate semantic information is introduced into model inputs.
- `hybrid_stats` Stage1 diagnostic became unstable: epoch 2 loss `3.35`, then epoch 3+ collapsed back to zero-pred scale (`loss ~= 6.7`, `pred_fr_norm=0`). Do not use this mode as mainline without stabilization/lower LR.

## 2026-07-09 Endpoint Semantic-Space KL Tilt v3
- Latest theory requirement: do not compute local continuation values and do not run ODE simulation inside training gradient steps. The clean chain is:
  `theta1 -> q_theta1(z|D) -> semantic pushforward -> semantic-space KL tilt toward y -> lifted q_plus(z|D) -> projected/cached theta1_plus -> supervised FM correction`.
- Implementation now uses objective version `endpoint_semantic_space_kl_tilt_v3` in graph and token endpoint-correction caches. This is important because older endpoint caches encode a different target objective and must not silently resume.
- The semantic tilt defaults to pure target semantic distance (`semantic_tilt_energy=target_distance`). Penalty-aware energy remains as an explicit ablation/fallback path, not the mainline definition of the Radon-Nikodym factor.
- Diagnostics now distinguish:
  - target-near mass lift: probability mass assigned to samples closest to target semantics should increase;
  - target-far mass suppression: probability mass assigned to far samples should decrease;
  - projected versions after mapping the lifted posterior back to `theta1_plus`.
- Centroid/mean semantic movement remains a diagnostic only. It should not be used as the hard acceptance criterion because multimodal SR posteriors can improve target-neighborhood mass while the first moment is ambiguous.
- Token branch smoke `smoke_token_endpoint_semantic_space_kl_v3_20260709` verified the target-neighborhood behavior:
  - target-near lift `1.3333`;
  - target-far suppression approximately `3.93e-17`;
  - projected target-near lift `4.0`;
  - projected target-far suppression `0.0`;
  - accept rate `1.0`.
- Static and regression checks passed after the v3 update:
  - `py_compile` for semantic mass utilities and graph/token/collector scripts;
  - `tests/test_semantic_mass_ng.py` passed 12 tests.
- This resolves the objective-definition mismatch for the semantic endpoint correction path, but it does not prove full SR success. Current full targets (`R2_mean > 0.95`, structure accuracy > 0.4 or clearly above baselines) remain open. Reference-field quality and full-scale endpoint-correction generalization still need gate validation.

## 2026-07-09 Dual-Branch v3 Medium Diagnostic
- Runner fix: the dual branch medium runner now explicitly uses `semantic_tilt_energy=target_distance` and ODE64 for both construction branches. This prevents the previous under-integration false negative on token rollout and avoids relying on implicit parser defaults.
- Token branch medium endpoint collection confirms the semantic-space KL tilt object behaves as designed at non-smoke scale:
  - target-near mass lift `2.0730`;
  - target-far mass suppression `0.0597`;
  - projected target-near lift `3.5156`;
  - projected target-far suppression `0.0234`;
  - endpoint accept rate `0.9375`.
- Token branch Stage2 is learnable in FM terms but not yet useful for target SR:
  - residual-only endpoint FM loss `0.3801`, cosine `0.9880`, norm ratio `0.9377`;
  - eval R2 `0.0406`, structure `0.0417`;
  - base syntax-prior eval R2 `0.0460`, structure `0.0`;
  - full-model train-base endpoint correction improves FM to `0.3351` but worsens eval R2 to `-0.0744`, structure `0.0`.
- Token diagnosis:
  - The semantic posterior/projection is not the immediate numerical failure; it does raise target-near semantic mass.
  - The current token parameterization/residual conditioning is not converting endpoint trace posterior supervision into a robust target-conditioned expression policy.
  - Since full-model fine-tuning does not improve eval, the next theoretical correction should focus on the projected endpoint target quality and construction-family conditioning, not merely unlocking more trainable parameters.
  - Candidate fixes should stay inside the semantic-space pushforward framework: richer complete-trace posterior statistics, better token construction masks/normal forms, and possibly a distillation target that preserves a posterior mixture over complete traces rather than collapsing through single rollout argmax behavior.
- Sampling coverage diagnosis:
  - Increasing token eval from 4 candidates/task to 32 candidates/task raises base R2 from `0.0460` to `0.2196` and v3 R2 from `0.0406` to `0.3489`.
  - Therefore the learned distributions do contain better semantic candidates than low-sample eval reveals.
  - v3 improves semantic coverage relative to base under the same 32-candidate budget, but structure remains essentially flat (`skeleton 0.0417`, `opdep 0.0`).
  - This points to a probability-concentration / structural-posterior problem, not just a terminal ODE or parser problem. Future correction should make the lifted posterior/projected endpoint preserve structural semantics or improve trace-family quality; simply sampling more is too expensive and does not meet the structural target.
- Graph branch medium Stage1 status after 8 epochs:
  - loss `1.1992`, cosine `0.9106`, norm ratio `0.7917`;
  - low-t loss remains high (`0-0.001=3.1070`, `0.001-0.005=3.2070`, `0.005-0.020=2.5732`) while high-t is much lower (`0.1-1.0=0.4706`).
  - Graph branch remains gated by reference velocity fitting; do not interpret graph Stage2 until Stage1 passes a sharpness/structure gate.

## 2026-07-09 Endpoint Semantic-Space KL Tilt v4
- The newest theory note keeps the same decoupled endpoint chain: flow once to obtain `theta1`, sample complete expressions from `q_theta1`, tilt the semantic pushforward distribution by pure target distance, lift to a complete-trace posterior, project/cache `theta1_plus`, then train the correction from the buffer without running ODE inside optimizer gradient steps.
- The required "mean statistic" should not be the raw semantic-vector centroid. A multimodal SR posterior can have an ambiguous centroid even when it correctly raises target-near mass and suppresses target-far mass.
- Implemented `target_near_far_contrast_*`: a signed semantic-space statistic with `+1` for target-near samples, `-1` for target-far samples, and `0` otherwise. Its mean improvement is exactly the desired top-near up / far-target down effect.
- Endpoint cache objective was bumped to `endpoint_semantic_space_kl_tilt_v4`, so older v3 endpoint buffers are rejected by signature mismatch.
- Acceptance now requires the signed contrast improvement to be positive in both ideal semantic posterior diagnostics and projected endpoint resampling diagnostics, in addition to target-distance improvement, near-mass lift, far-mass suppression, and top/bottom ratio lift.
- GPU token smoke confirmed the v4 fields are emitted from real endpoint collection/training:
  - objective version `endpoint_semantic_space_kl_tilt_v4`;
  - `semantic_endpoint_target_near_far_contrast_mean_improvement_mean = 0.1667`;
  - `semantic_endpoint_projected_target_near_far_contrast_mean_improvement_mean = 0.5`;
  - tiny-smoke accept rate `0.5`, as expected under a 2-group / 4-sample diagnostic.

## 2026-07-09 Endpoint Semantic-Signature KL Tilt v5
- Latest theory note keeps the decoupled endpoint chain and forbids optimizer-step ODE simulation:
  `theta1 -> q_theta1(z|D) -> semantic pushforward -> semantic-space KL tilt -> lifted q_plus -> projected theta1_plus -> supervised FM correction`.
- Implemented objective version `endpoint_semantic_signature_kl_tilt_v5` for both graph and token endpoint-correction caches. This rejects v4 buffers by metadata mismatch.
- The target-distance energy is no longer output-only normalized MSE. It is now a grouped semantic signature distance:
  - normalized output curve remains the primary term;
  - variable-sorted first/second-difference shape terms are added;
  - compact input/output correlation and dependency statistics are added.
- Output-only normalized MSE is retained only as `semantic_output_mse` / `semantic_endpoint_output_mse_*` diagnostics so we can tell whether a structural signature improvement is merely worsening raw fit.
- The grouped distance is intentionally not a flat MSE over a long signature vector. The curve term remains dominant, and shape/dependency terms add structure pressure without diluting target fit.
- Static verification passed after v5:
  `/home/ywj/miniconda3/envs/semflow/bin/python -m py_compile semflow_sr/semantic_mass.py scripts/train_complete_expression_semantic_fm.py scripts/train_token_policy_semantic_fm.py scripts/collect_semantic_mass_branch_metrics.py`
  and `/home/ywj/miniconda3/envs/semflow/bin/python -m pytest tests/test_semantic_mass_ng.py -q` (`13 passed`).
- GPU token smoke confirmed the v5 fields are emitted and the new semantic statistic moves in the desired direction:
  - run dir `/tmp/semflow_token_endpoint_v5_signature_smoke`;
  - `semantic_endpoint_objective_version=endpoint_semantic_signature_kl_tilt_v5`;
  - `semantic_endpoint_target_distance_mean_improvement_mean=4.6623`;
  - `semantic_endpoint_target_near_far_contrast_mean_improvement_mean=0.3334`;
  - `semantic_endpoint_projected_target_distance_mean_improvement_mean=4.6625`;
  - `semantic_endpoint_projected_target_near_far_contrast_mean_improvement_mean=0.5`;
  - ideal accept `1.0`, projected/overall accept `0.5`.
- Current interpretation: v5 fixes the "mean statistic" mismatch by making the reward tilt target a semantic signature that can raise target-near/top semantics and suppress far/unrelated semantics. It still needs medium/full validation; the full success criteria (`R2_mean > 0.95` and structure accuracy > `0.4` or clearly above baselines) are not yet met.
- Medium validation `token_endpoint_v5_signature_medium_20260709_r1` completed on GPU1:
  - Train/collection used 154 compiled train tasks, 128 endpoint groups, 2048 complete-trace examples.
  - Endpoint FM is learnable but not solved: best loss `0.3806`, cosine `0.9886`, norm ratio `0.9377`.
  - Semantic pushforward objective works in the intended direction:
    target-distance improvement `4.0876`, near/far contrast improvement `0.5059`, projected target-distance improvement `4.1060`, projected near/far contrast improvement `0.4795`, ideal accept `1.0`, projected/overall accept `0.75`.
  - Held-out eval remains poor: 40 tasks, `R2_mean=0.00398`, skeleton `0.0`, opdep `0.0`, valid expression fraction `0.78`, terminal max prob `0.875`.
  - Conclusion: the v5 semantic statistic and KL tilt are no longer the main immediate mismatch; the bottleneck is converting the tilted complete-trace posterior/projection into target-conditional structure recovery. Next grounded changes should target endpoint projection/trace posterior representation or the reference distribution, not add local continuation rewards or optimizer-step ODE.

## 2026-07-09 Endpoint Semantic-Signature Pushforward Projection v6
- The latest theory note keeps the same clean endpoint chain and still forbids local continuation values or ODE simulation inside optimizer gradient steps:
  `theta1 -> q_theta1(z|D) -> semantic pushforward -> semantic KL tilt -> lifted q_plus -> theta1_plus -> supervised FM correction`.
- v6 keeps the grouped semantic signature distance from v5, but adds a smoother target-rank utility diagnostic:
  - samples are ranked by pure grouped target distance;
  - the closest semantics get utility near `+1`, the farthest near `-1`;
  - the uniform-prior mean is centered at zero;
  - positive posterior/projected rank-utility improvement means probability mass moved toward target-near semantics across the whole sampled set, not only across a hard top/bottom quantile split.
- The cache objective version is now `endpoint_semantic_signature_pushforward_projection_v6`, so v5 and older endpoint buffers should not be reused silently.
- Acceptance now requires positive ideal and projected rank-utility improvement in addition to target-distance improvement, target-near mass lift, target-far suppression, concentration gain, contrast improvement, and top/bottom ratio lift.
- Token v6 smoke already passed on GPU0 before this note:
  - run dir `/tmp/semflow_token_endpoint_v6_rank_smoke`;
  - `semantic_endpoint_objective_version=endpoint_semantic_signature_pushforward_projection_v6`;
  - `semantic_endpoint_target_rank_utility_mean_improvement_mean=0.333438`;
  - `semantic_endpoint_projected_target_rank_utility_mean_improvement_mean=0.333333`;
  - `semantic_endpoint_target_near_far_contrast_mean_improvement_mean=0.333417`;
  - `semantic_endpoint_projected_target_near_far_contrast_mean_improvement_mean=0.25`;
  - `semantic_endpoint_tilt_accept_rate=0.5`.
- Current interpretation remains conservative: v6 verifies the endpoint semantic-statistic contract more directly, but it is still not evidence that full SR R2/structure targets are met. Full validation depends on the reference field and projection-to-policy bottlenecks identified in v5.

## 2026-07-09 Endpoint Semantic-Signature Pushforward Projection v7
- Latest theory note again confirms the desired chain is endpoint-decoupled:
  `theta1 -> q_theta1(z|D) -> semantic pushforward -> semantic KL tilt -> lifted q_plus -> theta1_plus -> supervised FM correction`.
- No active `train_complete_expression`, `train_token_policy`, `semantic_pushforward`, or `complete_expression_semantic_fm` process was running before the v7 edits; no unrelated GPU process was touched.
- v7 keeps the v6 grouped semantic signature and rank utility, but adds a continuous kernel-normalized target utility:
  - `target_soft_utility_mean_improvement > 0` means posterior mass increased a smooth target-neighborhood statistic;
  - `target_background_utility_reduction > 0` means the complementary unrelated/background semantic mass was suppressed.
- The same projected fields are recorded after resampling from `theta1_plus`. These projected soft utility fields remain diagnostic even when hard top/bottom neighborhoods are tied or ambiguous under small samples.
- Acceptance now also requires positive ideal/projected soft-utility improvement and background-utility reduction, while still requiring the hard projected neighborhood gap to be valid. This prevents reverting to a raw semantic centroid criterion and keeps the intended "top semantics up / unrelated semantics down" behavior explicit.
- Cache objective version was bumped to `endpoint_semantic_signature_pushforward_projection_v7`, so v6 endpoint buffers will not be reused silently.
- Static/unit validation passed:
  `/home/ywj/miniconda3/envs/semflow/bin/python -m py_compile semflow_sr/semantic_mass.py scripts/train_complete_expression_semantic_fm.py scripts/train_token_policy_semantic_fm.py scripts/collect_semantic_mass_branch_metrics.py`
  and `/home/ywj/miniconda3/envs/semflow/bin/python -m pytest tests/test_semantic_mass_ng.py -q` (`14 passed`).

## 2026-07-09 Endpoint Semantic-Signature Pushforward Projection v8
- Latest theory note keeps the same endpoint-decoupled chain and explicitly rejects training-step ODE simulation:
  `theta1 -> q_theta1(z|D) -> semantic pushforward -> semantic KL tilt -> lifted q_plus -> theta1_plus -> supervised FM correction`.
- v8 keeps v7 target soft utility, but fixes the background statistic:
  `target_background_utility_reduction` / `target_far_utility_reduction` now measure reduction in an independent semantic-distance far/background utility normalized on the prior sampled support.
- This is no longer merely `1 - target_soft_utility`; it separately checks that target-near kernel utility rises and far/unrelated semantic distance utility falls.
- Acceptance uses `target_far_utility_reduction` and `projected_target_far_utility_reduction` as the primary fields, with v7 `background_utility_reduction` retained as a compatibility alias.
- Cache objective version is now `endpoint_semantic_signature_pushforward_projection_v8`, so v7 endpoint buffers will not be reused silently.
- Static/unit validation passed:
  `/home/ywj/miniconda3/envs/semflow/bin/python -m py_compile semflow_sr/semantic_mass.py scripts/train_complete_expression_semantic_fm.py scripts/train_token_policy_semantic_fm.py scripts/collect_semantic_mass_branch_metrics.py tests/test_semantic_mass_ng.py`
  and `/home/ywj/miniconda3/envs/semflow/bin/python -m pytest tests/test_semantic_mass_ng.py -q` (`15 passed`).
- Tiny GPU token smoke verified the new fields are emitted:
  - run dir `/tmp/semflow_token_endpoint_v8_smoke`;
  - `semantic_endpoint_objective_version=endpoint_semantic_signature_pushforward_projection_v8`;
  - `semantic_endpoint_target_soft_utility_mean_improvement_mean=0.123720`;
  - `semantic_endpoint_target_far_utility_reduction_mean=0.125005`;
  - `semantic_endpoint_projected_target_soft_utility_mean_improvement_mean=0.207863`;
  - `semantic_endpoint_projected_target_far_utility_reduction_mean=0.207854`;
  - `semantic_endpoint_tilt_accept_rate=0.5`;
  - `semantic_endpoint_fm_loss=0.055339`.
- This validates the statistic and cache plumbing only. It does not change the prior conclusion that full SR success remains gated by reference-field quality and posterior/projection-to-policy structure recovery.

## 2026-07-09 Endpoint Semantic-Signature Pushforward Projection v9
- The newest theory note is already structurally aligned with the current Stage2 implementation: endpoint collection runs the flow once to get `theta1`, samples complete expressions from `q_theta1`, computes a semantic pushforward KL tilt, projects/lifts to `theta1_plus`, and trains correction from cached endpoints. There is no optimizer-step ODE simulation and no local continuation-value teacher in this path.
- The main missing piece was diagnostic/acceptance clarity: v8 separately tracked target-soft utility increase and independent far/background utility reduction. v9 combines them into a single signed mean statistic, `target_semantic_contrast_utility_mean_improvement`, where the utility is `target_soft_utility - target_far_utility`.
- This signed statistic matches the requirement better than a raw semantic centroid: it is defined on the sampled semantic distribution support and directly rewards raising target-near semantics while suppressing far/unrelated semantics. It remains compatible with multimodal SR posteriors because it does not force one semantic centroid.
- Acceptance still also checks hard top-near/far mass, rank utility, target-distance mean improvement, top/bottom mass ratio, and projected endpoint resampling. If ideal v9 statistics pass but projected statistics fail, the issue is posterior projection or construction parameterization, not the semantic KL tilt.
- v9 is an implementation/diagnostic correction only. It does not by itself solve the existing full SR bottleneck: previous medium runs showed semantic posterior statistics can be positive while R2/structure remain weak, so future validation must still inspect projection-to-policy and reference-field quality.

## 2026-07-09 v9 Validation Watch Points
- The v9 waiting controller is intentionally not evidence of algorithm success yet; it is only a safe launch mechanism. Training/eval begin only after GPU utilization thresholds are met.
- Primary v9 diagnostics to inspect once results are available:
  - `semantic_endpoint_objective_version` must equal `endpoint_semantic_signature_pushforward_projection_v9`;
  - `semantic_endpoint_target_semantic_contrast_utility_mean_improvement_mean > 0` checks the ideal lifted posterior raises target-top semantic utility and suppresses far utility;
  - `semantic_endpoint_projected_target_semantic_contrast_utility_mean_improvement_mean > 0` checks the projected endpoint preserves that effect;
  - `semantic_endpoint_tilt_accept_rate`, `projected_kernel_mass_lift_vs_prior`, and R2/structure metrics decide whether the bottleneck remains projection/reference/readout.
- Prior runs suggest the semantic posterior statistic can pass while R2/structure remain weak, so if v9 reproduces that pattern the next iteration should target posterior-to-policy/endpoint projection and reference-field coverage rather than adding local action losses.

## 2026-07-09 Endpoint Semantic-Signature Pushforward Projection v10
- User supplied a refined endpoint theory note emphasizing semantic-space metric guidance of parameter space, with collection after the flow endpoint and no ODE simulation inside optimizer gradient steps.
- Confirmed the existing endpoint chain was already aligned with the main theory:
  `theta1 -> q_theta1(z|D) -> semantic pushforward -> semantic KL tilt -> lifted q_plus -> theta1_plus -> supervised FM correction`.
- Stopped the waiting v9 controller before editing. It had not started graph/token training; logs and prior result directories were preserved.
- v10 keeps the same endpoint collection/training split and bumps the cache objective version to `endpoint_semantic_signature_pushforward_projection_v10`.
- Added `target_top_far_tail_utility`: a smoothed signed statistic defined on the prior sampled semantic support, with positive mass around target-near quantiles and negative mass around far-tail quantiles.
- Acceptance now requires positive ideal and projected top/far-tail utility improvement in addition to the existing target-distance, hard top/far, rank, soft target, far utility, signed target-vs-far, and top/bottom ratio checks.
- This is intentionally a diagnostic/acceptance tightening, not a new local continuation teacher or extra loss. It checks whether the sampled endpoint posterior actually raises target-top semantic probability and suppresses far/unrelated semantic tails.
- Static validation passed:
  `/home/ywj/miniconda3/envs/semflow/bin/python -m py_compile semflow_sr/semantic_mass.py scripts/train_complete_expression_semantic_fm.py scripts/train_token_policy_semantic_fm.py scripts/collect_semantic_mass_branch_metrics.py tests/test_semantic_mass_ng.py`.
- Unit tests passed:
  `/home/ywj/miniconda3/envs/semflow/bin/python -m pytest tests/test_semantic_mass_ng.py -q` (`16 passed`).
- No new GPU run was launched after v10 implementation; no matching SemanticFlowSR background training process remains active.

## 2026-07-09 v10 audit baseline
- Added a run-level audit layer because aggregate CSV fields were not enough to distinguish algorithmic failure modes.
- Important correction: train-only endpoint runs with empty `typed_op_node_flow_samples.jsonl` and `n_tasks=0` must not be treated as rollout/readout failures. They are missing eval evidence.
- Under the v10 objective version, historical v3-v9 and objective-unspecified runs mostly classify as `objective_version_mismatch`; this prevents using old positive posterior statistics as evidence for v10.
- Historical graph branch remains dominated by `reference_field` bottlenecks: Stage1 loss/cosine/norm ratio are still below the gate, so graph Stage2 should not be interpreted as full algorithm success until the reference field is repaired.
- Historical token branch often has a passable Stage1 FM loss but old endpoint objective versions and weak downstream eval. This supports the existing hypothesis: semantic posterior/projection can improve sampled semantic mass, but endpoint-to-policy/readout concentration and structure recovery remain unresolved.

## 2026-07-09 Endpoint Semantic-Signature Pushforward Projection v11
- The latest theory note re-emphasizes the clean endpoint chain:
  `theta1 -> q_theta1(z|D) -> semantic pushforward -> semantic KL tilt -> lifted q_plus -> theta1_plus -> supervised FM correction`.
- Stopped the pending v10 smoke waiting controller and audit watcher before editing. They had not started graph/token training because GPU utilization stayed above the configured threshold; no new v10 result directories were produced.
- v11 keeps the same endpoint-only collection/training split and bumps the cache objective version to `endpoint_semantic_signature_pushforward_projection_v11`.
- Added `target_top_tail_mass_contrast_mean_improvement`, the hard full-expression posterior mean statistic:
  `E_qplus[1_top_target - 1_far_tail] - E_q[1_top_target - 1_far_tail]`.
  This is the primary implementation of "raise target-near top semantic probability and weaken far/unrelated tail probability" and is deliberately not a semantic centroid or local continuation value.
- The smoother `target_top_far_tail_utility_mean_improvement` remains as an auxiliary/tolerance diagnostic. Acceptance now checks both the hard top-tail mass contrast and the smooth top/far-tail utility for ideal and projected endpoint distributions.
- Propagated v11 fields through graph endpoint correction, token weighted complete-trace FM, semantic-mass rollout diagnostics, progress rows, summaries, markdown output, metrics collection, and run audit.
- Updated `configs/eval/semantic_pushforward_gates.json` and the audit default expected objective to v11, so old v3-v10 results are classified as objective-version mismatches.
- Static/unit validation passed:
  `/home/ywj/miniconda3/envs/semflow/bin/python -m py_compile semflow_sr/semantic_mass.py scripts/train_complete_expression_semantic_fm.py scripts/train_token_policy_semantic_fm.py scripts/collect_semantic_mass_branch_metrics.py scripts/audit_semantic_pushforward_run.py tests/test_semantic_mass_ng.py`
  and `/home/ywj/miniconda3/envs/semflow/bin/python -m pytest tests/test_semantic_mass_ng.py -q` (`16 passed`).
- No new GPU training/smoke run was launched after v11 implementation; no SemanticFlowSR background training process remains active.

## 2026-07-09 Graph Target-Conditioned Stage1 Rebuild
- Token construction is no longer treated as the active path. Token training/eval runners, token train script, Stage2 endpoint/pushforward runners, branch collectors, old failed graph-gate artifacts, and related logs/results were moved into dated archive directories with manifests.
- Active graph Stage1 defaults are now target-conditioned:
  `training_flow=target_conditioned_reference`, `construction_graph=graph_dag_edge_simplex`, `task_conditioning=xy`, `theta0_endpoint_coupling=active_choice_bias`, `theta0_endpoint_bias=5.0`, `endpoint_target_mode=sampled_trace`, `reference_vector_field=bridge_path`, `reference_state_sampler=bridge_path`, and `semantic_action_features=false`.
- Added `eval_theta0_use_gt_trace=true` for evaluation seed coupling. This records whether eval used GT-trace-coupled `theta0`; it is a Stage1 construction diagnostic, not a Stage2 semantic guidance path.
- Added summary/report fields for raw R2 without affine fit, raw NMSE, expression degeneracy rates, eval-theta0 mode rates, GT variable-set coverage, and multi-variable GT prediction collapse rates. These fields force sample-level checks instead of relying only on aggregate R2.
- Smoke runner correction: `SYMBOLICGPT_TRAIN_LIMIT=0` in the loader means "unlimited", not "disabled". The smoke runner now leaves `SYMBOLICGPT_ROOT` empty by default, so the smoke uses a small benchmark train set unless SymbolicGPT is explicitly enabled.
- Representative GPU smoke `graph_target_conditioned_stage1_smoke_multivar_codex_20260709`:
  - Stage1 loss decreased from `0.2961` to `0.1431`, below zero-pred reference loss `0.6169`; cosine `0.9548`, norm ratio `1.1026`.
  - Eval over 8 benchmark tasks: `r2_mean=0.9218`, `solution_rate=0.625`, `skeleton_accuracy=0.75`, `operator_dependency_accuracy=0.75`.
  - Expressions were not constant/identity collapsed: `nontrivial_expression_rate=1.0`, `no_variable_expression_rate=0.0`, `identity_single_variable_expression_rate=0.0`.
  - Multi-variable coverage was present (`gt_multi_variable_task_rate=0.25`). Of the two multi-variable GT tasks, one recovered a multi-variable expression exactly up to affine scaling (`Jin-2: raw x0**2 + x1**3`), and one still collapsed to a one-variable approximation (`Jin-1: raw Abs(x0)**(1/4)`, R2 `0.4231`).
  - Interpretation: graph Stage1 is now viable enough to proceed to medium/full Stage1 validation, but multi-variable structure recovery is not fully solved. Do not claim Stage2 readiness from this smoke.
## 2026-07-10 Stage1 样例级审查

- 最近完成的 `graph_stage1_active_only_l6_medium_20260709` 在 48 个任务上
  `solution_rate=0`、`skeleton_accuracy=0`、`operator_dependency_accuracy=0`。
- samples 中表达式是可解析的随机合法树，而非接近 GT 的结构；多变量 GT 变量集合
  精确匹配率仅 0.0625。
- affine fit 制造了假高分：3 个样例 affine R2 > 0.9 但 raw R2 < 0。
- 实际配置为 `theta0_endpoint_coupling=none`，与既有闭环要求的
  active-choice coupling 直接冲突。
- 低 t loss 约 5.3-6.3，而 t>0.1 loss 为 0.2815；均匀采样导致整体 cosine 0.99
  掩盖 ODE 起点速度失败。
- GT trace probe 的 exact trace probability mean 仅 4.7e-5，经验一致样本率为 0。
- 2026-07-10 full e8 控制器仅通过 CUDA preflight，未生成 run 目录，当前无仓库训练进程。
- 核心 pytest 收集因缺失 `semflow_sr.edge_flow.pullback_chart` 失败。
- 完整诊断已追加到 `docs/STRUCTURAL_CLOSURE.md`。
