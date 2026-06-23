# Benchmark 数据集概览

当前评估通过统一 manifest 读取：

```text
data/benchmark_suites/benchmark_manifest.json
data/benchmark_suites/benchmark_index.csv
```

当前 Fisher / Euclidean 对比使用：

| suite | tasks | 用途 |
|---|---:|---|
| nguyen | 12 | 经典符号回归公式 |
| constant | 8 | 常数与单变量组合 |
| livermore | 8 | Livermore 公式 |
| jin | 6 | Jin 公式 |

总计：

```text
34 tasks
```

## Manifest Task 字段

每个 task 至少提供：

```json
{
  "task_id": "nguyen/Nguyen-1",
  "suite": "nguyen",
  "num_vars": 1,
  "variable_names": ["x0"],
  "train_path": ".../train.csv",
  "val_path": ".../val.csv",
  "test_path": ".../test.csv",
  "target_column": "target",
  "ground_truth": "x0**3 + x0**2 + x0"
}
```

`scripts/run_edge_flow.py` 会把不同维度任务 padding 到 checkpoint 支持的 `template.num_vars`，并保留 inactive-variable mask。

## 当前评估命令片段

```bash
--manifest data/benchmark_suites/benchmark_manifest.json
--manifest_root data/benchmark_suites
--manifest_suite nguyen constant livermore jin
```
