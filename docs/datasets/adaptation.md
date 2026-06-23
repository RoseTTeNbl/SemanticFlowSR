# 数据集适配

所有 benchmark task 会转成统一 `SRTask`，再交给 CSEF 和外部基线。

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

Manifest 中的 `variable_names` 统一使用：

```text
x0, x1, ...
```

## 主要模块

| 模块 | 作用 |
|---|---|
| `semflow_sr/data/benchmark_manifest.py` | manifest schema、读写、index |
| `semflow_sr/data/benchmark_loader.py` | `BenchmarkTaskSpec -> SRTask` |
| `semflow_sr/data/benchmark_validate.py` | split、列、finite 数值校验 |
| `semflow_sr/edge_flow/benchmark.py` | CSEF 评估 loader、padding、metrics 和结果写入 |
| `scripts/validate_benchmark_manifest.py` | manifest 校验 CLI |

## CSEF 评估接口

```bash
conda run --no-capture-output -n semflow python scripts/run_edge_flow.py \
  --ckpt checkpoints/teacher_path_geometry/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.pt \
  --out results/teacher_path_geometry_fisher_gpu \
  --tag teacher_path_geometry_fisher_gpu \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --manifest_root data/benchmark_suites \
  --manifest_suite nguyen constant livermore jin \
  --eval_samples 64 \
  --flow_steps 1 \
  --sampler_method policy \
  --head_fit_mode linear \
  --device cuda:1
```
