# 数据集文档

本目录记录当前 CSEF 使用的数据和 benchmark manifest。

## 当前数据源

训练数据：

```text
data/generated/symbolicgpt_subset
train: 747 formulas
val:   160 formulas
test:  161 formulas
```

评估数据：

```text
data/benchmark_suites/benchmark_manifest.json
data/benchmark_suites/benchmark_index.csv
```

当前结果使用的评估 suites：

```text
nguyen
constant
livermore
jin
```

总计 34 tasks。

## 文档

| 文档 | 内容 |
|---|---|
| [overview.md](overview.md) | manifest 结构和当前评估 suite |
| [deployment.md](deployment.md) | 数据生成、校验和本地路径 |
| [adaptation.md](adaptation.md) | `SRTask` 和评估脚本输入格式 |

## 常用命令

生成 SymbolicGPT-style 子集：

```bash
conda run --no-capture-output -n semflow \
  python scripts/generate_symbolicgpt_subset.py \
  --root data/generated/symbolicgpt_subset \
  --train_count 747 --val_count 160 --test_count 161 \
  --num_vars 3 --num_points 100 --max_depth 4
```

校验 benchmark manifest：

```bash
conda run --no-capture-output -n semflow \
  python scripts/validate_benchmark_manifest.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --root data/benchmark_suites \
  --out results/dataset_validation \
  --fail-on-error
```
