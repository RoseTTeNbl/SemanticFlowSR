# 修正记录与短板诊断

## 1.  为什么现在的“消融”反而更好

当前 `results/paper/` 里的消融组不是旧 plain Fisher 分支。它们都已经在新的
semantic-Fisher pullback 框架内：

```text
centered residual
-> action semantic effects xi
-> K = xi xi^T
-> semantic-Fisher log-rate target
-> sphere tangent matching
-> semantic_fisher_sphere / semantic_fisher_ode inference
```

当前 paper 表中：

| Row | 实际含义 |
|---|---|
| `Ours one-step reward` | 新 semantic-Fisher 主线，只把 scalar score provider 设为 one-step reward。 |
| `Ours one-step ODE` | 新 semantic-Fisher 主线，使用 ODE-style local update，但 score/provider 仍很短程。 |
| `Ours future ODE (no GP)` | 新 semantic-Fisher 主线，训练 target 使用 rollout-fitness provider；当前短预算下与 one-step ODE 结果相同。 |
| `GP as rollout policy` | GP 只作为 rollout completion policy 影响 future target。 |
| `GP policy distillation` | GP event likelihood 转成 online action/operator prior，加到 `w_theta`。 |

因此“现在消融之后反而更好”不是矛盾，而是名称造成的口径混淆：

```text
旧 0.15 = plain Fisher / no semantic pullback / potential endpoint
当前 one-step reward = semantic-Fisher pullback / lograte / one-step score provider
```

这两个实验检验的问题不同。

## 3. 当前 87-task 结果

数据集组成：

```text
Nguyen 12 + Constant 8 + Livermore 8 + Jin 6 + Feynman 53 = 87
1-var 21, 2-var 29, 3-var 37
```

主表：

| Group | Method | Mean R2 | Median R2 | Solution rate |
|---|---|---:|---:|---:|
| Baselines | PySR | 0.9974 | 1.0000 | 0.9310 |
| Baselines | DEAP | 0.9455 | 0.9960 | 0.3448 |
| Baselines | DSO | 0.9544 | 1.0000 | 0.5977 |
| SFSR ablations | Ours one-step reward | 0.9708 | 1.0000 | 0.7701 |
| SFSR ablations | Ours one-step ODE | 0.9505 | 1.0000 | 0.7126 |
| SFSR main | Ours future ODE (no GP) | 0.9505 | 1.0000 | 0.7126 |
| GP variants | GP as rollout policy | 0.9513 | 1.0000 | 0.7011 |
| GP variants | GP policy distillation | 0.8846 | 0.9417 | 0.3678 |

当前最强 SFSR 行是 `Ours one-step reward`，不是 future/GP 行。

## 4. 通过指标看到的主要短板

### 4.1 模型 ranking 仍然是第一短板

虽然最终 R2 不低，但局部 action rank 仍偏差明显。

| Method | selected rank mean | median | top-5 fraction | rank > 20 fraction |
|---|---:|---:|---:|---:|
| Ours one-step reward | 27.25 | 13 | 0.413 | 0.446 |
| Ours one-step ODE | 29.66 | 17 | 0.348 | 0.473 |
| Ours future ODE (no GP) | 29.66 | 17 | 0.348 | 0.473 |
| GP as rollout policy | 32.13 | 21 | 0.327 | 0.501 |
| GP policy distillation | 42.56 | 50 | 0.095 | 0.745 |

结论：

- 搜索成功很大程度上仍依赖多步过程和 affine readout 的纠错能力。
- 模型每步 `selected_probability_rank=1` 时，动作 reward rank 仍经常不是 top action。
- 下一步优化应继续围绕 action ranking，而不是只看 final R2。

### 4.2 Teacher / exact target 与模型之间仍有 gap

当前 exact target 的 rank 明显好于模型选择，说明模型还没充分学到 target field。

| Method | selected rank mean | exact semantic-Fisher top1 rank mean | exact top-5 fraction |
|---|---:|---:|---:|
| Ours one-step reward | 27.25 | 21.80 | 0.590 |
| Ours one-step ODE | 29.66 | 16.39 | 0.676 |
| Ours future ODE (no GP) | 29.66 | 16.39 | 0.676 |
| GP as rollout policy | 32.13 | 19.34 | 0.651 |
| GP policy distillation | 42.56 | 17.48 | 0.675 |

判断：

- exact semantic-Fisher target 不是完美 oracle，但通常比模型实际选择更好。
- 当前瓶颈不是单纯 `K` 或 solver 无效，而是 model imitation / generalization 不足。
- 应优先增加 model capacity、训练步数、hard-state mining、rank-aware auxiliary diagnostics，而不是先扩大 GP 分支。

### 4.3 ODE / future target 当前没有带来收益

`Ours one-step ODE` 与 `Ours future ODE (no GP)` 结果完全相同：

| Method | Mean R2 | Solution rate | one-step/rollout corr |
|---|---:|---:|---:|
| Ours one-step ODE | 0.9505 | 0.7126 | 0.7103 |
| Ours future ODE (no GP) | 0.9505 | 0.7126 | 0.7103 |

并且 rollout 相关诊断显示：

```text
one-step/rollout corr mean = 0.7103
low-corr (<0.3) fraction ~= 0.234
```

判断：

- 当前 rollout target 太弱，`eval_topk=1`、`max_completion_steps=1`、`n_rollouts_per_action=1` 只能提供非常浅的 future signal。
- 在约 76.6% 的 rollout steps 上，one-step 与 rollout score 仍高度一致或至少不冲突。
- 这解释了 future ODE 没有明显超过 one-step ODE。

后续建议：

1. 提高 rollout budget，只对 top-L + random subset 做 progressive rollout。
2. 把 `max_completion_steps` 从 1 提到 2/3，先离线缓存 target。
3. 记录 `rollout_rank_shift_abs_mean` 与 solved gain 的相关性，确认 future provider 是否真在纠正短视动作。

### 4.4 Feynman 是当前主失败来源

按 suite 看，SFSR 的弱项集中在 Feynman 和 Jin：

| Method | Nguyen | Constant | Livermore | Jin | Feynman |
|---|---:|---:|---:|---:|---:|
| Ours one-step reward | 0.917 | 0.625 | 0.875 | 0.667 | 0.755 |
| Ours one-step ODE | 0.917 | 0.875 | 0.875 | 0.667 | 0.623 |
| Ours future ODE (no GP) | 0.917 | 0.875 | 0.875 | 0.667 | 0.623 |
| GP as rollout policy | 0.917 | 0.875 | 0.875 | 0.667 | 0.604 |
| GP policy distillation | 0.917 | 0.750 | 0.875 | 0.500 | 0.094 |

判断：

- Nguyen / Livermore 基本已稳定。
- Jin 只有 6 个任务，波动大，但仍显示多元组合结构难点。
- Feynman 任务数量最多，直接决定总表；GP distillation 在 Feynman 上几乎崩溃，是总分下降的主因。

后续建议：

1. 分 suite 训练或至少分 suite 校准 operator prior。
2. 对 Feynman 单独检查 operator set、变量尺度、protected ops 与 complexity penalty。
3. 把 final expression complexity / active columns 加入主诊断表，避免高 R2 但表达式臃肿。

### 4.5 GP distillation 当前 prior 过粗，在线干预过强

GP distillation 行：

```text
solution_rate = 0.3678
Feynman solution_rate = 0.0943
selected rank mean = 42.56
top-5 fraction = 0.095
rank > 20 fraction = 0.745
gp_policy_applied = 495 / 495 steps
selected_gp_prior_rank_mean = 1.14
```

判断：

- GP prior 几乎每步都主导了动作选择。
- 当前 event extraction 主要是 operator-level prior，不含状态相似度、action slot、read/write context。
- 它会把局部语义上不合适的 operator 强行推到 top，尤其伤害 Feynman。

后续建议：

1. 降低 `gp_policy_weight`，做 `[0.05, 0.1, 0.2, 0.5]` 扫描。
2. GP distillation 必须加入 state-conditioned retrieval；只用 operator likelihood 不够。
3. 将 GP prior 从 additive hard bias 改成 gating / tie-breaker：只在模型 margin 小时启用。
4. 分 suite 蒸馏 GP prior，避免 DEAP-style operator frequency 迁移到 Feynman 时失真。

### 4.6 当前 best row 说明 one-step geometry 仍很强

`Ours one-step reward` 是当前 SFSR 最好行：

```text
Mean R2 = 0.9708
Solution rate = 0.7701
```

这说明 semantic-Fisher pullback 的局部几何修正本身是有效的。当前不应该把问题简单归因于“缺 GP”或“缺 rollout”。更直接的问题是：

```text
score provider / model ranking / GP prior calibration
```

还没有比 one-step semantic-Fisher local update 更稳。

## 5. 下一步优先级

### P0: 重新定义必要消融，避免口径混乱

必须同时保留两类消融：

| 消融 | 检验问题 |
|---|---|
| old plain Fisher potential / closed-form | 没有 `K` 和 semantic pullback 时会怎样；历史 0.15 属于这一类。 |
| current one-step reward | 在 semantic-Fisher 主线内，仅替换 score provider 为 one-step reward。 |

文档和图表中不要把这两类都叫 “one-step ablation”。

### P1: 提升 model imitation

目标：

```text
selected_reward_rank_mean < 10
top-5 fraction > 0.65
rank > 20 fraction < 0.2
```

建议：

- 增加训练 steps 和任务数，当前 paper configs 只有 40 steps / 32 tasks per dim。
- 做 hard-state replay：收集 `selected_reward_rank > 20` 的状态反复训练。
- 加 rank-aware diagnostics 或辅助 pairwise ranking loss 做 ablation，但主损失仍保留 sphere tangent matching。

### P2: 让 future provider 真正 future-aware

当前 rollout 太浅。下一版应优先使用 offline target cache：

```text
top-L first actions + random actions
max_completion_steps = 2 or 3
n_rollouts_per_action > 1
top-k mean aggregation
```

并把 `one_step_rollout_corr`、`rollout_rank_shift_abs_mean` 和 final solved gain 联合报告。

### P3: 重做 GP prior

当前 GP distillation 是 operator-level prior，只能作为负面诊断。下一版 GP 应至少包含：

- state embedding / residual similarity retrieval；
- action_id 或 action template 级别统计；
- suite-conditioned prior；
- margin-gated online injection。

### P4: 诊断表继续精简但要更稳

保留论文图：

- total R2 / solution rate；
- per-suite R2 / solution rate；
- action ranking；
- train loss / reward。

同时在 CSV 中保留：

- `selected_reward_rank_mean`
- `predicted_top1_reward_rank_mean`
- `exact_semantic_fisher_top1_reward_rank_mean`
- `one_step_rollout_corr_mean`
- `gp_policy_applied`
- final complexity / active columns

`support_best_reward_gap` 在数值极端 reward 下容易被尺度污染；后续更推荐报告 `full_best_in_support_rate` 和 clipped/median gap。
