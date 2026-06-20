# 数据集部署

所有命令在 `SemanticFlowSR/` 下执行。

## 1. 生成完整 benchmark

```bash
conda run -n semflow python scripts/prepare_benchmark_suites.py \
  --sources formula_dev srsd_main srsd_dummy pmlb \
  --pmlb-fetch-missing \
  --pmlb-limit 50 \
  --pmlb-max-samples 5000 \
  --pmlb-max-features 20 \
  --out-root data/benchmark_suites/materialized \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --index data/benchmark_suites/benchmark_index.csv
```

生成后目录应包含：

```text
data/benchmark_suites/benchmark_manifest.json
data/benchmark_suites/benchmark_index.csv
data/benchmark_suites/materialized/
```

## 2. 校验 manifest

```bash
conda run -n semflow python scripts/validate_benchmark_manifest.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --root data/benchmark_suites \
  --out results/dataset_validation \
  --fail-on-error
```

这一步必须在 SFSR 或 baseline 长跑前通过。

## 3. 生成 SFSR 评测命令

```bash
conda run -n semflow python scripts/run_sfsr_benchmark_matrix.py \
  --suite_group formula_dev srsd_main srsd_dummy pmlb \
  --ckpt_by_vars \
    1:checkpoints/d1.pt \
    2:checkpoints/d2.pt \
    3:checkpoints/d3.pt \
  --plan_out results/benchmark_plans/sfsr_risk_flow_commands.json
```

确认命令后加 `--execute`。正式完整评测需要补齐更高维 checkpoint；缺失维度会在
`*_skipped.json` 中记录。

## 4. 生成外部 baseline 命令

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --suite_group formula_dev srsd_main srsd_dummy pmlb \
  --plan_out results/benchmark_plans/external_baseline_commands.json
```

外部 baseline 的环境和预算在 `configs/eval/external_baselines.yaml` 里统一维护。
