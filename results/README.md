# Results Layout

This directory keeps the current CSEF experiment outputs.

## Current Result Directories

```text
teacher_path_geometry_fisher_gpu/
teacher_path_geometry_euclidean_gpu_20260623/
paper_metrics/csef_fisher_vs_euclidean_gpu_20260623/
```

`teacher_path_geometry_fisher_gpu/` contains the Fisher probability-shape run.
`teacher_path_geometry_euclidean_gpu_20260623/` contains the Euclidean probability-coordinate ablation.
`paper_metrics/csef_fisher_vs_euclidean_gpu_20260623/` contains normalized summary tables, paired significance, and figures for the two-line comparison.

## Expected Evaluation Files

Each evaluated method directory should contain:

```text
*_summary.json
*_samples.jsonl
*_task_expressions.csv
*_task_expressions.md
*_statistics_by_group.csv
*_statistics_by_group.json
*_diagnostics.json
```

## Expected Paper-Metrics Files

Each metrics directory under `results/paper_metrics/` should contain:

```text
method_summary.csv/json
suite_summary.csv/json
paired_significance.csv/json
metric_summary.png/pdf
pareto_r2_complexity.png/pdf
structural_metrics.png/pdf
manifest.json
```

Current headline metrics are recorded in `docs/RESULTS.md`.
