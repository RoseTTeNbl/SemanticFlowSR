# 数据集描述

当前数据策略仍分层，但 S1 的核心对象已经从 plain Fisher-sphere path sample 改成 semantic-Fisher local target。

## S1 层 — 局部 semantic-Fisher target dataset

这是当前基础训练的核心数据。

默认样本字段：

```python
{
  "x", "y", "B",
  "action_ids", "action_feats", "semantic_stats",
  "energies", "rewards", "advantages",
  "p_start", "p_target",
  "residual_current", "residual_next",
  "xi", "gram",
  "w_target", "pdot_target", "zdot_target",
  "lambda", "p_lambda", "dp_dlambda", "z_lambda", "dz_dlambda",
  "scores", "proposal_probs", "gt_action_pos"
}
```

其中语义-Fisher 主线下：

- `lambda = 0`
- `p_lambda = p_start`
- `dp_dlambda = pdot_target`
- `z_lambda = sqrt(p_start)`
- `dz_dlambda = zdot_target`

这些兼容字段还在，是为了旧测试和 ablation，不再是主理论对象。

关键增量字段：

- `xi`: 每个动作的 residual effect 向量
- `gram`: `xi @ xi.T`
- `semantic_stats`: 从 `xi`/`gram` 压缩出的 action-level 描述
- `w_target`: exact semantic-Fisher log-rate
- `zdot_target`: 主训练监督

生成链路：

```text
随机表达式树
-> 编译为寄存器轨迹
-> 对每个 trace step 采样 support
-> 计算 centered residual 与 action effects
-> provider 给分数 R(a)
-> 组标准化得到 A(a)
-> semantic_fisher_lograte 生成 target
```

## S2 层 — 公式 benchmark

公式库仍在 `configs/data/formula_benchmarks/*.yaml`：

- Nguyen
- Constant
- Livermore
- Jin

当前验证主结果使用 20 个 one-variable 子任务：

```text
Nguyen-1..8
Constant-1,2,5,6,8
Livermore-1,2,3,4,6,7,8
```

## A / B 层

- PMLB / SRBench：保留 benchmark loader 和部署脚本
- 更大规模 SRSD / NeSymReS：仍是后续扩展

它们不改变当前 local target 的几何定义。
