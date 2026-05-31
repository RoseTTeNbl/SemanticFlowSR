# 数据集适配

所有数据集最终通过 `semflow_sr/data/benchmark_loader.py` 统一成 `SRTask`，再被
`eval/evaluator.evaluate_task` 与各基线脚本消费。

## 统一接口 `SRTask`

```python
@dataclass
class SRTask:
    name: str
    X_train: np.ndarray        # [n_train, d]
    y_train: np.ndarray        # [n_train]
    X_test:  np.ndarray        # [n_test, d]
    y_test:  np.ndarray        # [n_test]
    expression: str | None     # 已知公式（PMLB 为 None）
    variable_names: list[str]  # 长度 d
    metadata: dict
```

## 列约定（所有 CSV / TSV）

`[特征列..., target]`：最后一列名为 `target`，其余为特征（变量）列。
公式基准用 `x0, x1, ...`；PMLB 沿用原始物理量名（如 `m_0, v, c`）。

## 三类来源的加载方式

| 来源 | 入口 | 说明 |
|---|---|---|
| 公式基准 YAML | `materialize_formula(entry, seed)` | sympy 求值采样，返回 `SRTask` |
| 物化 CSV | `pd.read_csv` + 手工组 `SRTask` | 见 README「阶段 3」示例 |
| PMLB | `PMLBLoader(root).load(name)` | 读 `datasets/<name>/<name>.tsv.gz`，按 `test_frac` 随机划分 |

## PMLB 适配要点（与本仓库实测一致）

`PMLBLoader.load(name)`：
1. 路径 `external/pmlb/datasets/<name>/<name>.tsv.gz`；
2. `pd.read_csv(sep='\t', compression='gzip')`；
3. `target` 列作 `y`，其余列作 `X`、列名作 `variable_names`；
4. 随机划分（默认 `test_frac=0.25`、`seed=0`）。

**已实测**：`feynman_I_10_7`（3 变量 `m_0,v,c`，75000/25000）、`feynman_I_6_2a`（1 变量
`theta`）、`cache_pmlb_subset.py --pattern feynman` 均正常。无 `.tsv.gz` 的目录
（如 strogatz、缺数据的 3 个 feynman）会触发 `FileNotFoundError`，被
`cache_pmlb_subset.py` 的 try/except 跳过。

## 把 PMLB 数据接入评估

PMLB 任务无已知端点分布，评估走与公式基准相同的 rollout 路径：

```python
import torch
from semflow_sr.data.benchmark_loader import PMLBLoader
from semflow_sr.eval.evaluator import evaluate_task
from semflow_sr.sr.ops import NAME_TO_ID
from semflow_sr.models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig

ck = torch.load("checkpoints/velocity_gt.pt", map_location="cpu", weights_only=False)
m = SemanticTransformer(SemanticTransformerConfig(d=3, K=8, hidden=128, row_layers=2, heads=4))
m.load_state_dict(ck["model"]); m.eval()

task = PMLBLoader("external/pmlb").load("feynman_I_10_7")
ops = [NAME_TO_ID[o] for o in ("add","sub","mul","sin","cos","square")]
rep = evaluate_task(m, task, K=8, ops_ids=ops, device=torch.device("cpu"),
                    max_steps=16, grid=5, greedy=True, max_support=128)
print(rep.r2, rep.expression)
```

> 注意：`SemanticTransformerConfig.d` 必须等于该任务的变量数（PMLB 各数据集不同），
> `K`/`ops_ids` 需与训练时一致。多变量任务建议用 `--num_vars` 匹配的 checkpoint。

## 新增数据源的适配清单

1. 整理成 `[特征..., target]` 的表格（CSV/TSV）；
2. 写一个返回 `SRTask` 的小 loader（或复用 `PMLBLoader` 的目录约定）；
3. 确认 `variable_names` 长度 = `X` 列数；
4. 用上面的 rollout 片段跑通一次评估即完成适配。
