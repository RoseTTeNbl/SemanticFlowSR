# 数据集部署

所有命令在 `SemanticFlowSR/` 下执行。

## 1. 生成训练子集

```bash
conda run --no-capture-output -n semflow \
  python scripts/generate_symbolicgpt_subset.py \
  --root data/generated/symbolicgpt_subset \
  --train_count 747 --val_count 160 --test_count 161 \
  --num_vars 3 --num_points 100 --max_depth 4
```

输出目录：

```text
data/generated/symbolicgpt_subset/train
data/generated/symbolicgpt_subset/val
data/generated/symbolicgpt_subset/test
```

## 2. 校验 Benchmark Manifest

```bash
conda run --no-capture-output -n semflow \
  python scripts/validate_benchmark_manifest.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --root data/benchmark_suites \
  --out results/dataset_validation \
  --fail-on-error
```

校验输出：

```text
manifest_validation_summary.json
manifest_validation_suites.csv
manifest_validation_tasks.csv
manifest_validation_failures.jsonl
```

## 3. 当前评估 Suite

当前结果只使用：

```text
nguyen constant livermore jin
```

如需扩展到更多 suite，必须在 [../RESULTS.md](../RESULTS.md) 中单独记录任务集合、checkpoint、评估预算和指标。
