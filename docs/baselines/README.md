# 外部基线说明

当前 CSEF 结果只包含 Fisher 与 Euclidean 两条本文方法线。外部符号回归方法用于后续论文对照，必须和本文方法使用相同 manifest 任务集合，或标记为原生协议参考。

## 已适配入口

```text
scripts/run_deap_baseline.py
scripts/run_gplearn_baseline.py
scripts/run_pysr_baseline.py
scripts/run_dsr_baseline.py
scripts/run_tpsr_manifest_baseline.py
scripts/run_local_diffusion_reference.py
```

矩阵配置：

```text
configs/eval/external_baselines.yaml
```

生成命令计划：

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --suite_group formula_dev \
  --plan_out results/benchmark_plans/external_baseline_commands.json
```

执行指定方法：

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --method gplearn DSO \
  --suite_group formula_dev \
  --execute
```

## 结果字段

每个 baseline task 记录：

```text
task_id, method, suite, domain, split, n_vars
r2_raw, r2_affine_refit, nmse, nmse_affine_refit
expression, complexity, runtime_sec
status, error_type, error
budget, ground_truth
```

## 论文指标输出

统一入口：

```bash
python scripts/archive_paper_metrics.py \
  --out results/paper_metrics/<tag> \
  --suite nguyen constant livermore jin \
  --method CSEF-Fisher SFSR sfsr_method samples_jsonl results/teacher_path_geometry_fisher_gpu/teacher_path_geometry_fisher_gpu_samples.jsonl \
  --method gplearn GP external_comparison baseline_json results/baselines_current/gplearn_formula_dev_seed0.json
```

`role` 取值：

```text
sfsr_method
external_comparison
native_protocol_reference
```

只有同一 manifest 上的逐任务输出进入公平比较主表；原生协议参考只做补充记录。
