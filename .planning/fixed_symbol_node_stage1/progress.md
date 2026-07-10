# Fixed Symbol Node Stage 1 Progress

## 2026-07-05

- Started new scoped plan after user requested pausing old mainlines and verifying Stage 1 velocity matching convergence.
- Process check found no active old semantic-control training process.
- Existing old plan `.planning/semantic_global_bridge_full_run` is complete and not suitable for the new fixed-symbol-node mainline.
- Removed stale KL-control check scripts and result/log directories:
  `scripts/check_{a,b,c}_*.sh`, `scripts/check_semantic_kl_teacher_rollout.py`,
  `scripts/collect_semantic_kl_control_bridge_metrics.py`,
  `scripts/run_semantic_kl_control_*.sh`, `logs/semantic_kl_control_bridge_checks_20260704`,
  and `results/.../semantic_kl_control_bridge_checks_20260704`.
- `python -m py_compile scripts/train_fixed_symbol_node_stage1.py` passed before route-coupling patch.
- Started fixed 64-trace overfit run; interrupted at epoch 16 because loss only fell from about 6.22 to 3.88.
- Added `--theta0-endpoint-coupling choice_bias`, `--theta0-target-bias`, and changed default `--time-sampling` to uniform for the fixed-symbol Stage 1 prototype.
- Added `--condition-on-theta0/--no-condition-on-theta0` and made Stage 1 model condition on the known initial route seed by default.
- Fixed eval rollout so it keeps `theta0` as the seed condition while integrating from a clone of that initial state.
- Seed-conditioned global model still plateaued near loss 0.49 by epoch 19.
- Patched `FixedSymbolVelocityNet` action head to include block-local current logits, seed logits, seed probability metadata, and a deeper head, while retaining global state context.
- Stronger route coupling diagnostics showed `theta0_target_bias=5.0` gives zero-prediction loss about 0.48 and `theta0_target_bias=6.0` gives about 0.18; strict convergence sanity used bias 6.0.
- Fixed-batch overfit result written to `results/clean_benchmark_20260701/ablations/fixed_symbol_node_stage1_20260705/runs/overfit32_seedcond_bias6_early_l8_v2`:
  `last_train_loss=0.049846`, `best_train_loss=0.049846`, `early_stopped=true`, `eval_active_argmax_match_mean=1.0`.
- Non-fixed 64-trace smoke result written to `results/clean_benchmark_20260701/ablations/fixed_symbol_node_stage1_20260705/runs/smoke64_seedcond_bias6_e60_l8`:
  `final_loss_mean=0.037525`, `last_train_loss=0.034936`, `best_train_loss=0.026363`, `eval_active_argmax_match_mean=1.0`.
- Rewrote docs around fixed symbol node Stage 1:
  `docs/ALGORITHM_COMPLETE_EXPRESSION_SEMANTIC_FM.md`,
  `docs/ARCHITECTURE_COMPLETE_EXPRESSION_SEMANTIC_FM.md`,
  `docs/MATH.md`,
  `docs/STRUCTURAL_CLOSURE.md`,
  `docs/DIAGNOSTIC_EXPERIMENTS_COMPLETE_EXPRESSION_FLOW.md`,
  and `docs/README.md`.
- Added reproducibility scripts:
  `scripts/run_fixed_symbol_node_stage1_overfit.sh` and `scripts/run_fixed_symbol_node_stage1_smoke.sh`.
- Final static checks passed:
  `python -m py_compile scripts/train_fixed_symbol_node_stage1.py`;
  `bash -n scripts/run_fixed_symbol_node_stage1_overfit.sh scripts/run_fixed_symbol_node_stage1_smoke.sh`.
- Added Stage 2 rollout-time energy guidance to `scripts/train_fixed_symbol_node_stage1.py`:
  complete-trace sampling, expression energy, self-normalized posterior marginals over active blocks,
  FR-capped probability correction, and base-vs-online rollout summaries.
- Smoke check passed on `/tmp/fixed_symbol_stage2_smoke` with `--rollout-guidance-mode both`.
- Cleaned additional old mainline residue configs/scripts/modules while preserving external baseline results.
- Added `scripts/run_fixed_symbol_node_stage2_full.sh`.
- Full-scale Stage 2 run launched in background:
  launcher PID `1807327`, Python child PID `1807574`,
  log `logs/fixed_symbol_node_stage2_20260705/full_l24_trace2048_e10_guidance.nohup.log`,
  output `results/clean_benchmark_20260701/ablations/fixed_symbol_node_stage2_20260705/runs/full_l24_trace2048_e10_guidance`.
- Initial full-run progress: `step 1 epoch=0 loss=0.489177`, `step 10 epoch=0 loss=0.383193`.
- Rough runtime estimate from early CPU progress: Stage 1 training around 3-3.5h; Stage 2 online guidance may add several more hours because it evaluates `256*64*32` sampled traces for the online branch.
- User requested running under the configured `semflow` GPU environment.
- Confirmed previous full run was indeed full configuration but was launched with base Python where CUDA was unavailable.
- Stopped the CPU run and restarted with:
  `PYTHON_BIN=/home/ywj/miniconda3/envs/semflow/bin/python`,
  `CUDA_VISIBLE_DEVICES=3`,
  `RUN_DEVICE=cuda:0`.
- Current GPU full run:
  launcher PID `2121171`, Python child PID `2121444`,
  same log/output paths, command line includes full config and `--device cuda:0`.
- GPU run first progress: `step 1 epoch=0 loss=0.441983`; `nvidia-smi` shows PID `2121444` on physical GPU 3.
- Full semflow GPU run completed on 2026-07-06 and wrote:
  `results/clean_benchmark_20260701/ablations/fixed_symbol_node_stage2_20260705/runs/full_l24_trace2048_e10_guidance`.
- Final Stage 1 full metrics:
  `final_loss_mean=0.024717`, `last_train_loss=0.027204`, `best_train_loss=0.007143`,
  `eval_active_target_prob_mean=0.994119`, `eval_active_argmax_match_mean=0.998429`.
- Stage 2 full metrics over 256 tasks:
  `stage2_off_energy_mean=0.022010`, `stage2_off_r2_mean=0.992388`,
  `stage2_online_semantic_energy_mean=0.022010`, `stage2_online_semantic_r2_mean=0.992388`,
  `stage2_online_minus_off_energy_mean=0.0`, `stage2_online_minus_off_r2_mean=0.0`.
- Guidance samples written: `fixed_symbol_stage2_guidance_samples.jsonl` with 512 rows
  (256 `off` + 256 `online_semantic`).
- User requested result files follow the `clean_linear_identity_e5` result schema with GT and structural metrics.
- Added `scripts/normalize_fixed_symbol_node_results.py`, independent of old `edge_flow` modules, to convert fixed-symbol Stage 2 outputs into:
  `typed_op_node_flow_samples.jsonl`, `typed_op_node_flow_results.md`,
  `typed_op_node_flow_summary.json`, `typed_op_node_flow_benchmark_records.csv`,
  `typed_op_node_flow_strict_method_summary.csv`, `typed_op_node_flow_eval_progress.json`,
  and `typed_op_node_flow_ode_sweep.jsonl`.
- Normalized full run metrics:
  `r2_mean=0.992388`, `solution_rate=0.988281`, `skeleton_accuracy=0.992188`,
  `simplified_symbolic_equivalence_rate=0.988281`, `operator_dependency_accuracy=0.988281`,
  `zero_expression_rate=0.0`, `single_term_rate=0.023438`.
- Updated `scripts/run_fixed_symbol_node_stage2_full.sh` so future full runs automatically call the normalizer after training.
