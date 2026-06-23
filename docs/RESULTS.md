# 实验结果记录

本文档只记录当前 CSEF 主线结果。

## 1. 当前结果范围

```text
method family: Conditional Semantic Edge Flow
date: 2026-06-23
device: cuda:1
train data: data/generated/symbolicgpt_subset, 747 train formulas
eval suites: nguyen constant livermore jin
eval tasks: 34
eval_samples: 64
flow_steps: 1
sampler_method: policy
head_fit_mode: linear
```

当前比较只包含两条几何线：

```text
CSEF-Fisher:    probability_path_geometry=fisher
CSEF-Euclidean: probability_path_geometry=euclidean
```

当前训练配置使用：

```text
teacher_target_mode=structural_denoising
target_shape_source=structural_denoising
```

## 2. Artifacts

Fisher:

```text
config:        configs/train/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.yaml
checkpoint:    checkpoints/teacher_path_geometry/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.pt
train curve:   checkpoints/teacher_path_geometry/train_curve_conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.csv
train log:     logs/csef_fisher_full_train_20260623.log
eval dir:      results/teacher_path_geometry_fisher_gpu
eval summary:  results/teacher_path_geometry_fisher_gpu/teacher_path_geometry_fisher_gpu_summary.json
eval samples:  results/teacher_path_geometry_fisher_gpu/teacher_path_geometry_fisher_gpu_samples.jsonl
```

Euclidean:

```text
config:        configs/train/conditional_edge_flow_gt_sampler_teacher_path_euclidean_gpu.yaml
checkpoint:    checkpoints/teacher_path_geometry/conditional_edge_flow_gt_sampler_teacher_path_euclidean_gpu.pt
train curve:   checkpoints/teacher_path_geometry/train_curve_conditional_edge_flow_gt_sampler_teacher_path_euclidean_gpu.csv
train log:     logs/csef_euclidean_full_train_20260623_rerun.log
eval dir:      results/teacher_path_geometry_euclidean_gpu_20260623
eval summary:  results/teacher_path_geometry_euclidean_gpu_20260623/teacher_path_geometry_euclidean_gpu_20260623_summary.json
eval samples:  results/teacher_path_geometry_euclidean_gpu_20260623/teacher_path_geometry_euclidean_gpu_20260623_samples.jsonl
```

Combined metrics output:

```text
results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623
results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623/method_summary.csv
results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623/suite_summary.csv
results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623/paired_significance.csv
results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623/metric_summary.png
results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623/pareto_r2_complexity.png
results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623/structural_metrics.png
```

## 3. Training Summary

| metric | Fisher | Euclidean |
|---|---:|---:|
| rows | 7470 | 7470 |
| optimizer steps | 939 | 939 |
| epoch range | 0..9 | 0..9 |
| device | cuda:1 | cuda:1 |
| batch_loss mean | 0.020235 | 0.030072 |
| semantic_teacher_loss_mean | 0.020292 | 0.030295 |
| semantic_teacher_loss nonzero mean | 0.027356 | 0.040841 |
| semantic_calibration_energy_mean | 0.126022 | 0.239929 |
| semantic_calibration_energy nonzero mean | 0.169895 | 0.323456 |
| semantic_teacher_trace_count mean | 20.6542 | 20.6542 |
| gt_neighborhood_compile_success overall | 0.6643 | 0.6643 |
| gt_neighborhood_compile_success nonzero rows | 0.8956 | 0.8956 |
| gt_neighborhood_compiled mean | 2.6700 | 2.6700 |

Interpretation:

```text
Fisher has lower teacher loss and lower semantic calibration energy.
Both geometries see the same GT-neighborhood coverage.
Euclidean produces larger calibrated velocity errors during training.
```

## 4. Evaluation Summary

| method | R2 mean | R2 median | R2 95% CI | NMSE mean | solution rate | exact skeleton | symbolic eq | op/dep | BLEU | token acc | edit dist | complexity | weighted complexity | valid rate | unique rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CSEF-Fisher | 0.937383 | 0.996123 | [0.880588, 0.981796] | 0.062617 | 0.3235 | 0.0294 | 0.0000 | 0.0294 | 0.1027 | 0.0692 | 19.0588 | 12.912 | 23.441 | 0.9899 | 0.9995 |
| CSEF-Euclidean | 0.940868 | 0.992659 | [0.894852, 0.979058] | 0.059132 | 0.2941 | 0.0000 | 0.0000 | 0.0000 | 0.0706 | 0.1275 | 18.0588 | 12.324 | 21.971 | 0.9775 | 1.0000 |

## 5. Suite Summary

| suite | Fisher R2 | Euclidean R2 | Fisher solution | Euclidean solution | Fisher complexity | Euclidean complexity |
|---|---:|---:|---:|---:|---:|---:|
| constant | 0.990889 | 0.985593 | 0.5000 | 0.3750 | 12.000 | 11.125 |
| nguyen | 0.983832 | 0.965745 | 0.4167 | 0.1667 | 12.917 | 12.417 |
| livermore | 0.984388 | 0.982102 | 0.2500 | 0.5000 | 11.875 | 11.500 |
| jin | 0.710469 | 0.776504 | 0.0000 | 0.1667 | 15.500 | 14.833 |

## 6. Paired R2 Significance

```text
n_matched: 34
mean_delta Fisher-Euclidean: -0.003486
median_delta Fisher-Euclidean: 0.000499
wins/losses/ties for Fisher: 19 / 14 / 1
sign_test_p: 0.486850
```

The current 34-task run does not show a significant R2 difference between Fisher and Euclidean paths.

## 7. Current Conclusion

```text
1. Both geometries train and evaluate successfully on GPU.
2. Euclidean has slightly higher mean R2.
3. Fisher has higher solution rate, BLEU, and valid expression fraction.
4. Fisher training loss and semantic calibration energy are lower.
5. Structural recovery remains very low under exact and audited structure metrics.
```

The main remaining issue is structural alignment. The current high R2 values still rely heavily on numerical fitting and sparse-head calibration; exact expression recovery is not solved. The next full GPU run should use the structural denoising teacher target in the current configs.
