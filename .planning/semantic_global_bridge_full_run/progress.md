# Progress

## 2026-07-03
- Started full train-test validation request.
- Located project git root at `/home/ywj/wyh/SFSR/SemanticFlowSR`.
- Created isolated planning record under `.planning/semantic_global_bridge_full_run`.
- Confirmed `global_bridge` and task split interfaces in `scripts/train_sparse_register_flow.py`.
- Ran CPU smoke accidentally with default Python; found PyTorch was CPU-only.
- Switched to `/home/ywj/miniconda3/envs/semflow/bin/python`, confirmed CUDA works with `CUDA_VISIBLE_DEVICES=3`.
- Ran GPU smoke successfully in `smoke_global_bridge_split_gpu_e1`.
- Started full-scale `symgpt700/eval50` run, then stopped the 20-epoch `base_e20` process because epoch 0 had not finished after about 8 minutes.
- Restarted a one-hour-target fast full-scale run: `benchmark_global_bridge_symgpt700_eval50_fast_e2`, PID `3705975`.
- Cleaned smoke result directories and failed/test logs, keeping only the active `fast_e2` log/pid.
- Integrated GT-active local semantic guidance into base global bridge teacher velocity: `target_v = bridge_v + eta * semantic_delta`.
- Disabled old group-sampling semantic improvement train/infer path; it now raises if requested.
- Ran smoke `smoke_gt_active_semantic_guidance_e1`; metrics showed nonzero GT-active semantic block count and delta norm, then removed the smoke result directory.
- Launched full-scale epoch-10 run `benchmark_global_bridge_gtactive_semantic_w1_e10`, PID `3794179`, with `gt_active_semantic_guidance_weight=1.0`.
- Epoch-10 run completed and wrote full outputs under `results/clean_benchmark_20260701/ablations/global_bridge_semantic_improvement_validation_20260703/runs/benchmark_global_bridge_gtactive_semantic_w1_e10`.
- Removed the obsolete no-semantic `benchmark_global_bridge_symgpt700_eval50_fast_e2` result and log to avoid result-directory confusion.
- Verified no train/eval task overlap and no residual training process.
