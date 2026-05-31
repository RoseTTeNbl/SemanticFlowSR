# 数据集部署

所有命令在 `SemanticFlowSR/` 下、`semflow` 环境中执行。

## S1：速度流轨迹数据集（本地生成）

```bash
python scripts/generate_trace_dataset.py \
  --num_tasks 2000 --num_vars 1 --max_depth 4 --K 8 --probe_size 128 \
  --target gt --max_support 128 \
  --out data/local_flow_traces/v0          # -> data/local_flow_traces/v0/traces.pt
```

关键参数：`--target {gt,semantic_oracle}` 选择端点 `p1`；`--K` 寄存器数；
`--max_support` 限制每步动作支撑集大小（GT 动作必保留）。
训练时也可让 trainer 在线构建数据集，无需先落盘。

## S2：公式基准（本地物化）

纯 sympy/numpy，无需联网下载。

```bash
python scripts/materialize_formula_benchmark.py \
  --suite nguyen constant livermore jin --seeds 0 1 2 3 4 \
  --out data/materialized
```

产出 `data/materialized/<suite>/<name>/seed_{k}_{train,test}.csv`。可用以下方式自检：

```bash
python -c "import pandas as pd,numpy as np; d=pd.read_csv('data/materialized/nguyen/Nguyen-1/seed_0_train.csv'); x=d['x0'].to_numpy(); print('误差',abs(d['target'].to_numpy()-(x**3+x**2+x)).max())"
# 误差应 ~1e-16
```

## A：PMLB / SRBench（本地已有数据）

> 上游 LFS 预算耗尽，`git lfs pull` 不可用。本仓库的 `external/pmlb/datasets/` 已由用户
> **手动补齐 Feynman 数据**（116/119 含数据），可直接缓存。

```bash
# 缓存全部含数据的 feynman（无数据的目录会被自动跳过）
python scripts/cache_pmlb_subset.py --pattern feynman --out data/pmlb/feynman

# 仅缓存前 N 个（冒烟）
python scripts/cache_pmlb_subset.py --pattern feynman --limit 5 --out /tmp/pmlb_test
```

产出 `data/pmlb/feynman/<name>/{train,test}.csv` + `metadata.json`（默认 75/25 划分）。

若将来要补齐 Strogatz 或缺失的 3 个 Feynman 数据文件，把对应的
`datasets/<name>/<name>.tsv.gz` 放入 `external/pmlb/datasets/<name>/` 即可，loader 自动识别。

## 验证已部署数据的适配性

```bash
python -c "
from semflow_sr.data.benchmark_loader import PMLBLoader
ld = PMLBLoader('external/pmlb')
t = ld.load('feynman_I_10_7')
print(t.name, 'Xtr', t.X_train.shape, 'vars', t.variable_names)
"
# feynman_I_10_7 Xtr (75000, 3) vars ['m_0', 'v', 'c']
```
