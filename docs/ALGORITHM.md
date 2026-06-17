# 算法实现

当前主线是一条 semantic-Fisher pullback flow：

```text
centered semantic energy
-> action semantic effect xi(a)
-> semantic-Fisher pullback metric
-> exact local log-rate target w*(a)
-> sphere tangent z_dot*
-> model predicts w_theta
-> semantic_fisher_sphere / semantic_fisher_ode update
```

当前实现支持两个层级：

- **single local step**: 在 `p_start` 上预测一次 `w_theta` 并做一次 sphere retraction。
- **local ODE mode**: 在固定局部状态 `c=(B,y,S)` 内重复更新 `p_tau`，每个中间点都把当前 `p_tau` 输入模型。

它学习的不是普通动作分类器，也不是任意速度场，而是局部 action simplex 上的更新算子。每个训练样本都绑定一个条件

```text
c = (B, y, S, p_start)
```

其中 `S` 是当前 support，`p_start` 是这个 support 上的起点策略。训练和推理都必须显式传入 `action_ids`、`p_start`、`semantic_stats` 和 `gram`，不能把不同 support 上的分布混用。

## 1. 局部语义对象

### 1.1 条件变量

| 符号 / 字段 | 含义 | 代码位置 |
|---|---|---|
| `B` | 当前寄存器语义矩阵，shape `[m, K]` | `registers/executor.py` |
| `y` | 当前 probe target，shape `[m]` | dataset / benchmark loader |
| `S` | 当前候选动作 support | `actions/support_sampler.py` |
| `action_ids` | support 内动作编号，shape `[A]` | `trace_dataset.py` |
| `p_start` | support 上归一化起点分布，shape `[A]` | `endpoints/base.py` |
| `proposal_probs` | support sampler 的 proposal 概率 | `actions/support_sampler.py` |

`m` 是 probe 点数，`K` 是 register 数，`A=|S|` 是当前 support 大小。

### 1.2 Centered energy

全链路使用同一个 centered ridge projection backend：

```text
C = I - 1/m 11^T
e(B) = Cy - Pi_{CB,rho} Cy
E(B) = 1/2 ||e(B)||^2
```

代码位置：

- `semflow_sr/semantics/projection.py::ProjectionBackend.residual_vector`
- `semflow_sr/semantics/projection.py::ProjectionBackend.residual_energy`
- `semflow_sr/semantics/energy.py::ActionEnergy`

所有 one-step reward、rollout fitness、residual feature 和 evaluation energy 都应使用这个 backend。

### 1.3 动作分数与语义效果

一步动作后得到 `B^a`，对应 residual：

```text
e_c      = Cy - Pi_{CB,rho} Cy
e_c^a    = Cy - Pi_{CB^a,rho} Cy
xi_c(a)  = e_c - e_c^a
K_c(a,b) = <xi_c(a), xi_c(b)>
```

动作标量 reward：

```text
R(a) = E(B) - E(B^a) - lambda_op C_op(a)
```

动作向量效果 `xi_c(a)` 记录 residual 如何移动；Gram 矩阵 `K_c` 记录动作之间的语义效果相似度。当前主线的关键区别是：target 不只看 scalar advantage，还用 `K_c` 扭曲 Fisher 几何。

代码里 `ActionEnergy.action_semantic_effects` 返回：

| 字段 | shape | 含义 |
|---|---:|---|
| `residual_current` | `[m]` | 当前 centered residual `e_c` |
| `residual_next` | `[A, m]` | 每个动作后的 residual `e_c^a` |
| `xi` | `[A, m]` | residual improvement vector |
| `gram` | `[A, A]` | `xi @ xi.T` |
| `rewards` | `[A]` | 与同一 residual backend 对齐的一步 reward |

## 2. Target 构造流程

训练 target 由 `VelocityTraceDataset` 动态构造。主路径 `path_name="semantic_fisher_pullback"` 下，单个样本的基础构造流程是：

```text
1. evaluate current registers:
   B = eval_register_state(state, x)

2. enumerate full valid actions:
   A_full = action_space.valid_actions(state)

3. score full actions with centered energy:
   full_rewards = ActionEnergy.evaluate_actions(B, y, A_full)

4. sample support:
   S = SupportSampler.sample(A_full, rewards=full_rewards, gt_action_id=...)

5. compute support-local reward and semantic effects:
   rewards, energies = evaluate_actions(B, y, S)
   residual_current, residual_next, xi, gram = action_semantic_effects(B, y, S)

6. build start policy:
   p_start = prior.build_p0(B, y, S, context)

7. build scalar target scores:
   scores = provider(B, y, S)
   advantages = group_standardize(scores)

8. solve semantic-Fisher log-rate:
   w_target = semantic_fisher_lograte(p_start, advantages, gram, beta, gamma)

9. convert to sphere tangent and one-step target endpoint:
   z0 = sqrt(p_start)
   zdot_target = 1/2 z0 * w_target
   p_target = semantic_fisher_sphere_step(p_start, w_target, dt=1)
```

实现位置：

- `semflow_sr/data/trace_dataset.py::VelocityTraceDataset`
- `semflow_sr/flow/semantic_fisher.py::semantic_fisher_lograte`
- `semflow_sr/flow/semantic_fisher.py::semantic_fisher_sphere_step`

### 2.1 多时间点 teacher path

如果配置：

```yaml
flow_training:
  train_along_path: true
  num_time_samples: 2
  target_integration_steps: 2
```

dataset 会先用 exact semantic-Fisher teacher field 在固定 `(A,K)` 上积分：

```text
p_0 = p_start
for i in 0..L-1:
    w_i = SFLogRate(p_i, A, K, beta, gamma)
    p_{i+1} = SemanticFisherSphereStep(p_i, w_i, dt=1/L)
```

然后采样一个中间策略 `p_tau`，并在这个中间点重新求：

```text
w_tau = SFLogRate(p_tau, A, K, beta, gamma)
z_dot_tau = 1/2 sqrt(p_tau) * w_tau
```

此时 batch 里：

| 字段 | 含义 |
|---|---|
| `p_start` | 局部 teacher path 的原始起点 |
| `p_lambda` | 本次监督点的当前策略 `p_tau` |
| `lambda` | `tau` 的归一化位置 |
| `w_target` | 在 `p_tau` 处重新解出的 exact log-rate |
| `zdot_target` | 在 `sqrt(p_tau)` 处的 sphere tangent |

这避免了推理多步 ODE 时偏离训练流形：模型训练时已经见过中间策略位置。

### 2.2 Advantage 标准化

provider 只负责给 `scores`。当前支持：

| Provider | 分数来源 | 用途 |
|---|---|---|
| one-step | centered energy decrease | 基础训练主线 |
| rollout | first action 后的 completion fitness 聚合 | 未来价值 target |
| search | beam / search after first action | search-improvement target |
| GP interface | GP-induced score 或 policy-derived score | 只保留接口 |

统一转换为 group advantage：

```text
A(a) = (R(a) - mean_S R) / (std_S R + eps)
```

若配置启用 clipping，则记录 clip 后的 `advantages`。主线 target 用的是这个 `A(a)` 和当前 `gram`，而不是直接训练 endpoint 分类。

### 2.3 Semantic-Fisher log-rate

给定 `p=p_start`，`P=diag(p)`，`K=gram`：

```text
M = I + gamma K P
M w* = beta (A + nu 1)
p^T w* = 0
```

其中 `nu` 用质量守恒约束确定。实现会做数值修正，保证：

```text
sum_a p(a) w*(a) ~= 0
```

含义：

- `beta`: 更新强度，同时控制 target log-rate 尺度。
- `gamma`: semantic pullback 权重。`gamma=0` 时退化为普通 Fisher replicator。
- `K`: 动作效果 Gram，控制语义相近动作之间的几何耦合。
- `w*`: support-local log-rate，不是概率，也不是 logits 分类标签。

## 3. 模型输入与输出

默认模型：

```python
SemanticTransformer(output_mode="semantic_fisher_lograte")
```

输入包括：

| 输入 | shape | 作用 |
|---|---:|---|
| `x, y, B` | `[m,d]`, `[m]`, `[m,K]` | row/register semantic context |
| `action_ids` / `action_feats` | `[A]`, `[A,F]` | 动作类型、读写槽位、复杂度等静态特征 |
| `energies` | `[A]` | 动作后 energy |
| `weights` | `[A]` | 兼容字段，主线通常为 1 |
| `p_start` / `p_lambda` | `[A]` | path 起点 / 当前监督或推理策略 |
| `semantic_stats` | `[A,8]` | 从 `xi` 和 `gram` 压缩出的动作语义统计 |
| `gram` | `[A,A]` | action-relation mixing |
| `action_mask` | `[A]` | padding 后的合法动作 mask |

输出：

| 输出 | 含义 |
|---|---|
| `lograte_logits` | 中心化后的 `w_theta(c,a)` |
| `v_pred` | `p_start * w_theta`，用于诊断 |
| `z_dot_pred` | `1/2 sqrt(p_start) * w_theta`，用于主损失 |

`lograte_logits` 会按当前输入分布中心化。单步训练时当前分布是 `p_start`；path 训练和 ODE 推理时当前分布是 `p_lambda`。它满足：

```text
sum_a p_current(a) w_theta(a) ~= 0
```

## 4. 主训练目标

主损失是 sphere tangent matching：

```text
L_SF = ||z_dot_theta - z_dot_target||^2
```

其中：

```text
z_dot_theta = 1/2 sqrt(p_current) * w_theta
z_dot_target = 1/2 sqrt(p_current) * w_target
```

实现位置：

- `semflow_sr/train/losses.py::SemanticFisherVelocityLoss`
- `semflow_sr/train/trainer_velocity.py::train_velocity`

训练循环简化为：

```text
for batch in loader:
    out = model(
        B=batch["B"],
        y=batch["y"],
        action_feats=batch["action_feats"],
        p_lambda=batch["p_lambda"],
        lambda_value=batch["lambda"],
        semantic_stats=batch["semantic_stats"],
        gram=batch["gram"],
        action_mask=batch["action_mask"],
    )

    loss = SemanticFisherVelocityLoss(
        p_start=batch["p_lambda"],
        w_target=batch["w_target"],
        w_pred=out.lograte_logits,
        zdot_target=batch["zdot_target"],
        z_dot_pred=out.z_dot_pred,
    )
```

### 4.1 保留的必要消融

只保留两类消融：

| 消融 | 配置 / 入口 | 目的 |
|---|---|---|
| no-pullback | `gamma=0` | 检查 semantic Gram 是否带来收益 |
| plain Fisher potential | `loss_name="sphere_path"` 或 `integration_method="closed_form"` | 对照无 `K` 的 exponential Fisher 更新 |

其他历史分支不再作为主实验路径维护。

## 5. 推理流程

主推理入口：

- `semflow_sr/search/rollout_velocity.py::rollout_velocity`
- `scripts/run_experiment.py`

每个 SR step 的流程：

```text
1. evaluate current B and energy E(B)
2. enumerate full legal action set
3. compute full one-step rewards for diagnostics
4. sample support S
5. compute support rewards, xi, gram, semantic_stats
6. build p_start
7. model predicts w_theta on S
8. update p by semantic_fisher_sphere_step
9. select action by argmax or sampling
10. execute selected action and continue
```

主更新公式：

```text
z = sqrt(p_start)
z_next = normalize_positive(z + dt * 1/2 z * w_theta)
p_next = z_next^2
```

默认配置：

```text
integration_method = semantic_fisher_sphere
step_size = 1.0
num_policy_updates = 1
```

多步 ODE 推理使用：

```text
integration_method = semantic_fisher_ode
ode_steps = 2 or 4
step_size = 1 / ode_steps
```

每个 ODE sub-step 都把最新的 `p` 输入模型：

```text
for j in 0..ode_steps-1:
    w_j = model(c, p_j, tau_j)
    p_{j+1} = SemanticFisherSphereStep(p_j, w_j, dt)
```

`num_policy_updates > 1` 会在同一个离散 SR step 内重复局部 policy update：

```text
p^(j+1) = SemanticFisherSphereStep(p^(j), w_theta(c, p^(j)))
```

目前默认只做一次，避免把多次局部更新误当作全局搜索能力。

### 5.1 Future reward rollout

`target=rollout_fitness_advantage` 会把 scalar score 从 one-step reward 替换成短 horizon completion score：

```text
Q_H(c,a) = Aggregate_j Fitness(rollout_j after first action a)
A_H(c,a) = GroupNormalize(Q_H(c,a))
w_H = SFLogRate(p, A_H, K, beta, gamma)
```

当前实现支持：

| 参数 | 含义 |
|---|---|
| `max_completion_steps` | 首动作后的 completion 深度 |
| `n_rollouts_per_action` | 每个首动作 rollout 次数 |
| `rollout_policy` | `random` / `semantic_greedy` / `mixed` / `gp_guided` |
| `reward_aggregation` | `mean` / `max` / `topk_mean` |
| `eval_topk` | 只对 one-step top-k 动作做 rollout，其余用 fallback |
| `complexity_penalty` | 动作 op-cost penalty |
| `final_complexity_penalty` | completion 后最终表达式复杂度 penalty |

### 5.2 GP-guided rollout policy

GP 当前不改变主 solver。它作为 rollout completion policy 的先验：

```text
rollout_policy = gp_guided
gp_action_scores[id] or gp_operator_scores[op] -> choose preferred completion action
```

`gp_action_scores` 适合事件回放；`gp_operator_scores` 适合跨状态的轻量先验，例如偏向 `mul/square/protected_div`。这只是 rollout policy 引导，不是 GP target 主链路。

## 6. 训练样本字段

默认 local target record 包含：

```text
x, y, B
action_ids, action_feats, action_mask
semantic_stats
energies, rewards, scores, advantages
one_step_rewards, rollout_rewards
proposal_probs
residual_current, residual_next, xi, gram
gamma
p_start, p_target
w_target, pdot_target, zdot_target
plain_p_target
gt_action_pos, full_action_size
```

兼容字段仍保留：

```text
lambda, p_lambda, dp_dlambda, z_lambda, dz_dlambda, p0, p1
```

在 semantic-Fisher 主线下：

| 字段 | 值 |
|---|---|
| `lambda` | `0` |
| `p_lambda` | `p_start` |
| `dp_dlambda` | `pdot_target` |
| `z_lambda` | `sqrt(p_start)` |
| `dz_dlambda` | `zdot_target` |
| `p0` | `p_start` |
| `p1` | `p_target` |

开启 `train_along_path` 后，`p_lambda` 改为中间策略 `p_tau`，`w_target / zdot_target` 也在 `p_tau` 处重新计算。

## 7. 指标含义

### 7.1 主损失与局部拟合

| 指标 | 来源 | 含义 | 趋势 |
|---|---|---|---|
| `semantic_fisher_velocity_loss` | `SemanticFisherVelocityLoss` | `z_dot_pred` 与 `zdot_target` 的 MSE | 越低越好 |
| `endpoint_kl` | loss diagnostics | 由 `w_pred` 和 `w_target` 各自 sphere step 后的 `KL(p_target || p_pred)` | 越低越好 |
| `lograte_corr` | loss diagnostics | `w_theta` 与 `w_target` 的 masked Pearson-like 相关 | 越高越好 |
| `lograte_top1_agreement` | loss diagnostics | `argmax w_theta` 是否等于 `argmax w_target` | 越高越好 |
| `pred_top1_reward_rank_mean` | loss / trainer diagnostics | 模型 top-1 动作在真实 reward 中的平均排名 | 越接近 1 越好 |

`semantic_fisher_velocity_loss` 低只说明切向量拟合好；真正决定搜索质量的是 ranking 指标，尤其是 `pred_top1_reward_rank_mean`。

### 7.2 分布更新强度

| 指标 | 含义 | 诊断用途 |
|---|---|---|
| `p_start_entropy` | 更新前 support 分布熵 | 起点策略是否过尖或过平 |
| `p_target_entropy` | target sphere step 后分布熵 | exact target 的保守程度 |
| `pred_endpoint_entropy` | model update 后分布熵 | 模型更新是否过激或过弱 |
| `kl_p_target_p_start` | target update 相对起点的 KL | target 步长大小 |
| `kl_p_pred_p_start` | model update 相对起点的 KL | 模型实际步长大小 |
| `l1_p_target_p_start` | target update 的 L1 距离 | target 分布移动幅度 |
| `l1_p_pred_p_start` | model update 的 L1 距离 | 模型分布移动幅度 |
| `p_target_top1_mass` | target top-1 概率质量 | target 集中程度 |
| `p_final_top1_mass` | 推理最终分布 top-1 概率质量 | 实际选择置信度 |

常见判断：

- `kl_p_pred_p_start` 远大于 `kl_p_target_p_start`: 模型更新过激。
- `kl_p_pred_p_start` 接近 0: 模型没有学会有效更新。
- entropy 很低但 reward rank 很差: 模型自信地选错动作。

### 7.3 Ranking 指标

| 指标 | 含义 | 诊断用途 |
|---|---|---|
| `pred_top1_advantage_agreement` | 模型 top-1 是否等于 target log-rate top-1 | 学 target 排序能力 |
| `pred_top1_advantage_rank_mean` | 模型 top-1 在 `w_target` 排名中的平均名次 | 是否接近 exact semantic-Fisher target |
| `pred_top1_reward_rank_mean` | 模型 top-1 在 reward 排名中的平均名次 | 是否选到局部好动作 |
| `selected_reward_rank` | 推理实际选中动作的 reward rank | rollout 中最重要的局部质量指标 |
| `selected_probability_rank` | 选中动作在 `p_final` 中的概率排名 | argmax 时应为 1 |
| `selected_advantage_rank` | 选中动作在 normalized advantage 中的排名 | provider target 是否被遵守 |
| `exact_semantic_fisher_top1_reward_rank` | exact `w_target` top-1 在 reward 中的排名 | target 本身是否合理 |
| `plain_fisher_top1_reward_rank` | plain Fisher 对照 top-1 在 reward 中的排名 | semantic pullback 是否改善局部排序 |

如果 `exact_semantic_fisher_top1_reward_rank` 好而 `pred_top1_reward_rank_mean` 差，主要问题在模型学习；如果 exact target 自己也差，问题在 reward/provider 或 support。

### 7.4 Reward 与 support 指标

| 指标 | 含义 | 诊断用途 |
|---|---|---|
| `full_action_size` | 当前完整合法动作数 | 搜索空间规模 |
| `support_size` | 实际 support 大小 | 模型每步处理的动作数 |
| `support_mode` | support sampler 模式 | 对比 full/topk/random/mixed |
| `full_best_reward` | full action set 中最好 reward | oracle 局部上界 |
| `support_best_reward` | support 中最好 reward | support 质量 |
| `support_best_reward_gap` | `full_best_reward - support_best_reward` | support 是否漏掉最好动作 |
| `full_best_in_support` | full best 是否进入 support | support coverage |
| `reward_mean/std/min/max` | support 内 target score 统计 | target 数值尺度 |
| `advantage_min/max` | normalized advantage 范围 | 是否被 clipping 或退化 |
| `proposal_prob_min/max` | support proposal 概率范围 | proposal 偏置 |
| `correction_weight_max` | `1 / proposal_prob` 最大值 | importance correction 风险 |
| `importance_ess` | proposal correction 有效样本数 | 越低说明权重越集中 |

`support_best_reward_gap` 长期为 0 但 selected rank 差，说明主要不是 support 漏动作，而是模型或 target 排序问题。

### 7.5 Rollout / search target 指标

这些指标只在 rollout/search provider 生成的 batch 中出现：

| 指标 | 含义 |
|---|---|
| `one_step_reward_mean/std` | one-step reward 统计 |
| `rollout_reward_mean/std` | rollout 聚合 score 统计 |
| `one_step_rollout_corr` | one-step reward 与 rollout score 的相关 |
| `one_step_rollout_top1_agreement` | 两者 top-1 是否一致 |
| `rollout_eval_fraction` | support 中实际被 rollout 评估的比例 |
| `rollout_rank_shift_mean` | rollout rank 相对 one-step rank 的平均变化 |
| `rollout_rank_shift_abs_mean` | rank shift 的平均绝对值 |
| `rollout_best_score_max/mean` | rollout evaluated 动作中的 best score 统计 |
| `rollout_best_final_energy_min` | rollout completion 后最低 final energy |
| `rollout_best_final_r2_max/mean` | rollout completion 后 R2 统计 |

若 `one_step_rollout_corr` 很低，说明 one-step target 短视；这时 rollout/search provider 才有明确价值。

### 7.6 Path / update 诊断

开启 `--record_path` 后，每个推理 step 会记录更新轨迹摘要：

| 字段 | 含义 |
|---|---|
| `update` | 第几次 local policy update |
| `ode_step` | `semantic_fisher_ode` 内部子步编号 |
| `lambda` | 记录点。semantic-Fisher 主线通常记录更新前后摘要 |
| `p_entropy` | 当前分布熵 |
| `p_top1_mass` | 当前 top-1 概率质量 |
| `velocity_norm` | 模型诱导 tangent / log-rate 相关范数 |
| `velocity_abs_max` | 最大绝对更新分量 |
| `tangent_error` | 质量守恒误差诊断 |
| `update_kl` | 本次 update 相对 update start 的 KL |
| `update_distance` | 本次 update 的 L1 距离 |
| `beta` | 当前更新强度 |
| `integration_method` | `semantic_fisher_sphere` 或保留消融 |

`tangent_error` 应接近 0；如果明显变大，优先检查 mask、padding 或分布归一化。

### 7.7 最终 SR 结果指标

| 指标 | 含义 |
|---|---|
| `r2` | 最终 centered/affine readout 后的 R2 |
| `nmse` | normalized mean squared error |
| `solved` | 是否达到当前 evaluator 的 solved 阈值 |
| `steps` | rollout 使用的动作步数 |
| `complexity` | 最终 active expression complexity |
| `simplicity` | complexity 派生的简洁度分数 |
| `energy_trace` | 每步 centered residual energy |
| `active_columns` | 最终 affine readout 使用的 register |
| `readout_coefficients` | 最终线性 readout 系数，最后一项为 intercept |

高 `r2` 不一定等于符号表达式紧凑；需要同时看 `complexity`、`active_columns` 和最终表达式。

## 8. 推荐评测流程

### 8.1 基础正确性

```bash
conda run -n semflow pytest -q
```

重点覆盖：

- centered projection / residual energy
- action semantic effects
- semantic-Fisher log-rate solver
- sphere step positivity and normalization
- dataset target fields
- inference update helpers

### 8.2 基础训练

```bash
conda run -n semflow python -m semflow_sr.train.train_base_natural_flow \
  --config configs/train/base_natural_flow.yaml
```

先看：

- `semantic_fisher_velocity_loss`
- `lograte_corr`
- `pred_top1_reward_rank_mean`
- `kl_p_pred_p_start` vs `kl_p_target_p_start`

### 8.3 标准评测

```bash
conda run -n semflow python scripts/run_experiment.py \
  --ckpt checkpoints/velocity_one_step_advantage.pt \
  --suite nguyen constant livermore jin \
  --seed 0 \
  --out results/semantic_fisher \
  --tag formula_1var_seed0 \
  --max_steps 12 \
  --grid 1 \
  --step_size 1.0 \
  --max_support 32 \
  --support_mode mixed_topk_random \
  --support_topk 16 \
  --target one_step_advantage \
  --integration_method semantic_fisher_sphere \
  --beta 1.0 \
  --gamma 0.1 \
  --record_diagnostics \
  --record_path \
  --device cpu
```

优先看：

1. `solution_rate`
2. `mean R2` / `median R2`
3. `selected_reward_rank`
4. `predicted_top1_reward_rank`
5. `support_best_reward_gap`
6. `plain_fisher_top1_reward_rank`

### 8.4 必要消融

| 消融 | 做法 | 判断 |
|---|---|---|
| `gamma=0` | 保持主线 solver，只把 pullback 权重设 0 | 验证 semantic Gram 是否有效 |
| plain Fisher endpoint | `--integration_method closed_form`，配合 potential checkpoint | 验证旧 exponential Fisher 分布更新的差距 |
| support size | 扫 `max_support` / `support_topk` | 判断是否 support-limited |
| rollout target | `target=rollout_fitness_advantage` | 判断 one-step reward 是否短视 |
| GP-guided rollout | `target_kwargs.rollout_policy=gp_guided` | 判断 GP-style prior 是否改变 completion policy |

## 9. Proximal / GP 扩展边界

rollout、search、GP 都只改变 scalar score provider：

```text
scores -> group-normalize -> A(a) -> semantic_fisher_lograte -> sphere update
```

它们不改变：

- centered residual energy
- action semantic effect `xi`
- Gram `K`
- semantic-Fisher linear system
- sphere retraction update

`gp_distill/` 和 `targets/gp_implicit_target.py` 目前只保留接口，不进入基础训练主链。
