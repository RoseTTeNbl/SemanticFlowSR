# 修正记录

## 这轮改了什么

旧主线是：

```text
scalar advantage
-> plain Fisher-sphere path
-> potential path matching
-> closed-form endpoint inference
```

现在主线是：

```text
centered residual
-> action semantic effects xi
-> semantic-Fisher pullback metric
-> exact local log-rate target
-> sphere tangent matching
-> semantic_fisher_sphere inference
```

## 代码层面的关键变化

- 新增 `ProjectionBackend.residual_vector`
- 新增 `ActionEnergy.action_semantic_effects`
- 新增 `semflow_sr/flow/semantic_fisher.py`
- 模型默认输出改为 `lograte_logits`
- 主损失改为 `SemanticFisherVelocityLoss`
- 默认数据集路径改为 `semantic_fisher_pullback`
- 默认推理改为 `integration_method=semantic_fisher_sphere`
- `gram` 与 `semantic_stats` 进入模型编码

旧 `sphere_path` / `closed_form` / `potential` 分支仍保留，但只作为 ablation 或兼容入口。

## 为什么必须这样改

plain Fisher-sphere 版本的主要问题不是分布不合法，而是 local ranking 很差。历史结果：

- `solution_rate = 0.15`
- `selected_reward_rank_mean = 23.625`
- `pred_top1_reward_rank_mean = 24.5`

也就是说，模型几乎总在按自己的 top-prob 动作走，但这个动作通常不是高 reward 动作。

问题根源是旧目标只有标量 advantage，没有动作语义效果之间的几何。

## 当前验证结果

### 单元测试

当前全量：

```text
84 passed, 1 warning
```

新覆盖包括：

- centered residual vector
- semantic effect / Gram extraction
- semantic-Fisher solver
- `gamma=0` 退化到 plain Fisher
- sphere tangent legality
- semantic-Fisher dataset record
- semantic-Fisher loss零误差情形

### 训练

`configs/train/base_natural_flow.yaml`

- final `semantic_fisher_velocity_loss = 0.0013967`
- held-out `reward(r2) = 0.9282`

### 评测

`results/semantic_fisher/formula_1var_seed0_summary.json`

- `mean R2 = 0.999041`
- `median R2 = 0.999999`
- `solution_rate = 0.95`

局部动作指标：

- `selected_reward_rank_mean = 3.65`
- `pred_top1_reward_rank_mean = 3.65`
- `selected_probability_rank_mean = 1.0`
- `exact_semantic_fisher_top1_reward_rank_mean = 6.4`
- `plain_fisher_top1_reward_rank_mean = 15.4`

## 当前判断

这次修正不是只把 loss 换了个名字。训练、推理和理论对象现在一致，而且实际 ranking 指标和 benchmark 结果都明显改善。

仍然留下的工作主要是：

1. 复杂度仍偏高，表达式坍缩到紧凑真值的能力还要继续查。
2. rollout/search/GP 现在只保留接口，应该在 one-step 稳定后再接回主实验。
3. 论文 PDF 还没有同步到 semantic-Fisher pullback 版本，仓内 markdown 已先更新。
