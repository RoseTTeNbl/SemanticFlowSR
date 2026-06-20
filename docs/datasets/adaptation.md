# 数据集适配

所有数据集先转成统一 `SRTask`，再交给 SFSR 和外部 baseline。

```python
@dataclass
class SRTask:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    expression: str | None
    variable_names: list[str]
    metadata: dict
```

## CSV 约定

每个 split 是普通 CSV：

```text
x0,x1,...,target
```

PMLB 原始列名会保存在 metadata，但 manifest 里的 `variable_names` 统一为
`x0, x1, ...`，方便 SFSR checkpoint 按维度选择。

## 主要模块

| 模块 | 作用 |
|---|---|
| `semflow_sr/data/benchmark_manifest.py` | manifest schema、读写、index |
| `semflow_sr/data/benchmark_loader.py` | `BenchmarkTaskSpec -> SRTask` |
| `semflow_sr/data/benchmark_validate.py` | 全量校验 split、列、finite 数值 |
| `scripts/prepare_benchmark_suites.py` | 物化 Formula / SRSD-Feynman / PMLB |
| `scripts/validate_benchmark_manifest.py` | 写验证报告 |

## 生成或重修 manifest

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

然后立刻校验：

```bash
conda run -n semflow python scripts/validate_benchmark_manifest.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --root data/benchmark_suites \
  --out results/dataset_validation \
  --fail-on-error
```

## SFSR 矩阵

SFSR 不需要按 suite 手写命令。先生成命令计划：

```bash
conda run -n semflow python scripts/run_sfsr_benchmark_matrix.py \
  --suite_group formula_dev srsd_main srsd_dummy pmlb \
  --ckpt_by_vars \
    1:checkpoints/d1.pt \
    2:checkpoints/d2.pt \
    3:checkpoints/d3.pt \
  --plan_out results/benchmark_plans/sfsr_risk_flow_commands.json
```

确认后加 `--execute`。如果 checkpoint map 缺少某个维度，`run_experiment.py` 会跳过
对应任务并写 `<tag>_skipped.json`；正式主表建议补齐 d1 / d2 / d3 / d4+ checkpoint。

当前矩阵方法：

```text
SFSR-RiskFlow-H1
SFSR-RiskFlow-H3
SFSR-RiskFlow-H5
SFSR-RiskFlow-FullSelector
```

## 外部 baseline 矩阵

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --suite_group formula_dev srsd_main srsd_dummy pmlb \
  --plan_out results/benchmark_plans/external_baseline_commands.json
```

确认环境后执行指定方法：

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --method DEAP gplearn PySR DSO \
  --suite_group formula_dev \
  --execute
```

说明：主环境中的 `deap` / `gplearn` 安装服务于 GP-assisted SFSR 工具和 smoke。
作为论文 baseline 时，它们仍按 `configs/eval/external_baselines.yaml` 的独立方法记录。

## GP Population 文件

GP-CandidatePool / GP-PriorReplacement / GP-PosteriorLikelihood 需要 action trajectory：

```json
{"actions": [12, 31, 44], "gp_logprob": -4.2, "fitness": 0.98}
```

加载入口：

```python
from semflow_sr.gp_distill.trajectory_pool import load_gp_trajectory_population
```
