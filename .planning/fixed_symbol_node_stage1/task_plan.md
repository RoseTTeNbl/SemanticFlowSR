# Fixed Symbol Node Stage 1 Plan

## Objective

Pause obsolete semantic-control mainlines, keep only a clean fixed-symbol-node Stage 1 reference-flow prototype, and verify whether its velocity matching loss can converge before adding semantic guidance.

## Phases

| Phase | Status | Notes |
| --- | --- | --- |
| 1. Pause and cleanup old mainline residue | complete | Removed stale KL-control check scripts and result/log directories; preserved comparison baselines and datasets. |
| 2. Check theory/code alignment | complete | Identified independent theta0/endpoint coupling as the main source of ambiguous low-t targets. |
| 3. Repair Stage 1 implementation | complete | Added route coupling, theta0 conditioning, block-local features, and early-stop loss. |
| 4. Verify velocity matching overfit | complete | Fixed-batch overfit reached last/best loss 0.049846 with active argmax match 1.0. |
| 5. Run small Stage 1 validation | complete | Non-fixed 64-trace smoke reached final loss mean 0.037525 and best loss 0.026363. |
| 6. Update docs | complete | Rewrote current algorithm, architecture, math, closure, diagnostic, and docs README around fixed-node Stage 1. |
| 7. Implement Stage 2 energy guidance | in_progress | Add complete-trace energy posterior estimation and online guidance rollout. |
| 8. Run full-scale Stage 2 validation | complete | Full semflow GPU run completed and wrote Stage 1/Stage 2 summaries. |

## Constraints

- Do not delete `results/clean_benchmark_20260701/ablations/clean_boundary_20260702`.
- Do not revive semantic-gradient, collocation mixture, endpoint-shape, denoising, group sampling, or inactive prior supervision.
- Inactive/default coordinates must have loss weight zero.
- Stage 1 is syntax/reference-flow first; semantic online correction is not part of this validation.
- Preserve datasets and externally useful baselines.
- Stage 2 may use complete-trace sampled rewards at rollout time; it must not use GT intermediate semantics.

## Results

| Run | Key Result |
| --- | --- |
| `overfit32_seedcond_bias6_early_l8_v2` | `last_train_loss=0.049846`, `best_train_loss=0.049846`, `eval_active_argmax_match_mean=1.0` |
| `smoke64_seedcond_bias6_e60_l8` | `final_loss_mean=0.037525`, `last_train_loss=0.034936`, `best_train_loss=0.026363`, `eval_active_argmax_match_mean=1.0` |
| `full_l24_trace2048_e10_guidance` | `final_loss_mean=0.024717`, `stage2_off_r2_mean=0.992388`, `stage2_online_semantic_r2_mean=0.992388` |

## Active Full Run

```text
pid_file: logs/fixed_symbol_node_stage2_20260705/full_l24_trace2048_e10_guidance.pid
log: logs/fixed_symbol_node_stage2_20260705/full_l24_trace2048_e10_guidance.nohup.log
out: results/clean_benchmark_20260701/ablations/fixed_symbol_node_stage2_20260705/runs/full_l24_trace2048_e10_guidance
```

Configuration:

```text
num_layers=24
output_terms=3
trace_count=2048
epochs=10
stage2_tasks=256
stage2_ode_steps=64
online_guidance_samples=32
rollout_guidance_mode=both
```
