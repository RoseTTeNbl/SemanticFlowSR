# Benchmark 数据集概览

当前统一 benchmark 已经物化到 CSV split，并由一个 manifest 管理：

```text
data/benchmark_suites/benchmark_manifest.json
data/benchmark_suites/benchmark_index.csv
data/benchmark_suites/materialized/<suite>/<task>/{train,val,test}.csv
```

当前 manifest 规模：

| 分组 | Suites | Tasks | 用途 |
|---|---|---:|---|
| dev | Nguyen / Constant / Livermore / Jin | 34 | workflow、debug、旧公式结果对齐 |
| main | SRSD-Feynman easy / medium / hard | 120 | scientific discovery 主表 |
| main | PMLB regression filtered | 50 | 黑盒回归、鲁棒性、runtime |
| appendix | SRSD-Feynman dummy easy / medium / hard | 120 | dummy variable / feature selection |

总计：

```text
324 tasks = 34 dev + 170 main + 120 appendix
```

## Manifest 字段

每个任务至少包含：

```json
{
  "task_id": "srsd_feynman_hard/feynman-i.30.3",
  "suite": "srsd_feynman_hard",
  "num_vars": 4,
  "variable_names": ["x0", "x1", "x2", "x3"],
  "train_path": "materialized/.../train.csv",
  "val_path": "materialized/.../val.csv",
  "test_path": "materialized/.../test.csv",
  "target_column": "target",
  "ground_truth": "optional sympy string",
  "domain": "physics",
  "metrics": ["r2", "nmse", "complexity"]
}
```

## 验证命令

每次重修数据集后先跑：

```bash
conda run -n semflow python scripts/validate_benchmark_manifest.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --root data/benchmark_suites \
  --out results/dataset_validation \
  --fail-on-error
```

输出：

```text
manifest_validation_summary.json
manifest_validation_suites.csv
manifest_validation_tasks.csv
manifest_validation_failures.jsonl
```

这一步检查 split 文件是否存在、变量列和 target 是否齐全、数据是否 finite，并统计
每个 suite / 维度的任务数。

## Suite Groups

评测矩阵统一使用这些组：

```text
formula_dev
srsd_main
srsd_dummy
pmlb
all_complete
```

它们定义在：

```text
configs/eval/sfsr_full_benchmark.yaml
configs/eval/external_baselines.yaml
```
