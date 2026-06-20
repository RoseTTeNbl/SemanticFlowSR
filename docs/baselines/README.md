# 外部 Baseline 说明

本仓库把两类东西分开：

1. **SFSR 内部 GP 工具**：主 `semflow` 环境里的 `deap` / `gplearn` 主要用于
   `GP-CandidatePool`、`GP-PriorReplacement`、`GP-PosteriorLikelihood` 这三组
   GP-assisted SFSR 实验，或做轻量 smoke。
2. **论文外部对比 baseline**：DEAP、gplearn、PySR、DSO/DSR 等作为独立方法跑，
   有自己的预算、环境、结果文件和失败记录。

不要把第一类工具的 smoke 结果直接当作论文 baseline 表。

## 统一入口

所有已适配的 baseline 脚本都支持统一 manifest：

```text
scripts/run_deap_baseline.py
scripts/run_gplearn_baseline.py
scripts/run_pysr_baseline.py
scripts/run_dsr_baseline.py
```

统一矩阵配置在：

```text
configs/eval/external_baselines.yaml
```

先生成命令计划，不执行：

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --suite_group formula_dev srsd_main pmlb \
  --plan_out results/benchmark_plans/external_baseline_commands.json
```

确认环境和预算后执行指定方法：

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --method PySR DSO \
  --suite_group formula_dev \
  --execute
```

`TPSR` 目前作为 native protocol reference 保留在配置中，默认不执行。它使用
`external/TPSR/run.py` 的原生 PMLB/Feynman protocol，不接收本仓库 manifest 参数。

## 结果字段

`run_baseline_records` 会为每个任务写一条记录，失败不会中断整套数据集：

```text
task_id, method, suite, domain, split, n_vars
r2_raw, r2_affine_refit, nmse, nmse_affine_refit
expression, complexity(如果方法返回), runtime_sec
status, error_type, error
budget, ground_truth
```

主表至少报告：

```text
R2_raw / R2_affine_refit / runtime_sec / complexity / failed task count
```

SFSR 使用 affine readout，因此外部方法也保留 `r2_affine_refit`，避免只比较原始表达式
时口径不一致。

## Sanity Check

正式对比前先跑简单任务：

```bash
conda run -n semflow python scripts/check_baselines_sanity.py \
  --baseline gplearn \
  --generations 20 \
  --population_size 500 \
  --out results/baseline_sanity/gplearn
```

如果 `y=x`、`y=x^2`、Nguyen-1/2 都失败，先修 adapter 或预算，不进入主表。
