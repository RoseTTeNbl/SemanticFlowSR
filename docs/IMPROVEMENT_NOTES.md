# 改进记录：action-flow 主线的问题、指标和下一步

本文档记录 action-flow 主线的实测问题和后续改进建议。旧 87-task 数字来自
PathPosterior-Frequency target，是历史 baseline；semantic-projection 版本也是
上一轮理论改写，不是当前主线的新结果。

当前主线是：

```text
Semantic-Fisher Flow Matching
```

即：

```text
prefix state
-> deterministic action support A_s
-> deterministic p_init
-> TargetSampler builds q_hat(a|s)
-> lambda-dependent log q_hat - log p_lambda
-> semantic effects xi(a) define geometry only
-> semantic-Fisher endpoint ODE
-> flow matching
-> 推理时 commit 一个 action 或 STOP
```

已废弃的历史路线包括：

```text
PathPosterior-Frequency: risk-weighted visited-action frequency q*(a|s)
Terminal Semantic Projection: terminal residual direction -> semantic projection q*(a|s)
```

---

## 1. 当前实现状态

已经落地：

```text
action 作为唯一主生成单位；
虚拟 STOP action；
推理和训练的数值健康过滤；
每个 state 的 support budget；
多轮 on-policy trajectory resampling；
deterministic p_init with step-dependent STOP bias；
TargetSampler endpoint q_hat；
OneStepTarget 实验组；
FutureGroup-L3Target 实验组；
CachedTrajectoryFitnessTarget 可选实验组；
GPCandidateFitnessTarget 可选实验组；
ImportanceSamplingTarget / MCMCShapeTarget 可选实验组；
lambda-dependent endpoint log-ratio；
训练/推理 support cap 一致；
teacher path 上多个 p_lambda 的 flow matching record；
CPU thread limit；
dataset build / target sampler timing diagnostics；
```

关键代码：

```text
semflow_sr/path_posterior/action_support.py
semflow_sr/path_posterior/sampler.py
semflow_sr/path_posterior/dataset.py
semflow_sr/train/train_path_posterior_flow.py
scripts/run_path_posterior_flow.py
configs/train/semantic_fisher_flow_87_one_step.yaml
configs/train/semantic_fisher_flow_87_future_group_l3.yaml
```

---

## 2. 当前训练配置

当前非 smoke 配置：

```text
configs/train/semantic_fisher_flow_87_future_group_l3.yaml
```

主要参数：

```text
num_tasks = 16
target_mode = future_group_l3
num_trajectories = 8
max_states_per_task = 4
on_policy_iterations = 2
steps_per_iteration = 40
max_steps = 6
K = 10
rollout_depth = 3
rollouts_per_action = 1
max_rollout_support = 8
teacher_steps = 2
max_support_size = 32
enable_stop = true
max_abs_semantic = 1e6
max_energy_growth = 100
torch_num_threads = 4
torch_num_interop_threads = 1
```

以下训练日志属于旧 frequency-target 87-task run，保留作 baseline：

```text
iter 0 step 0   loss 0.034816
iter 0 step 20  loss 0.050343
iter 0 step 40  loss 0.038733
iter 0 step 60  loss 0.039471
iter 1 step 80  loss 0.044101
iter 1 step 100 loss 0.009531
iter 1 step 120 loss 0.018311
iter 1 step 140 loss 0.016703
iter 2 step 160 loss 0.009958
iter 2 step 180 loss 0.006318
iter 2 step 200 loss 0.005752
iter 2 step 220 loss 0.004095
iter 3 step 240 loss 0.008077
iter 3 step 260 loss 0.004410
iter 3 step 280 loss 0.006006
iter 3 step 300 loss 0.011457
```

checkpoint：

```text
checkpoints/semantic_fisher_flow_future_group_l3_87.pt
```

---

## 3. 已清理的旧 frequency-target 87-task 记录

旧 `PathPosterior-Frequency` 结果文件已从当前结果目录清理；以下数字只作为历史
诊断说明，不再作为当前实验入口或可复现实验 tag。

```text
n_tasks = 87
skipped = 0
mean R2 = 0.7229
median R2 = 0.8709
mean NMSE = 0.2771
median NMSE = 0.1291
solution_rate@R2>=0.999 = 0.0
mean complexity = 38.85
median complexity = 39.0
mean steps = 6.0
median steps = 6.0
energy_decrease_mean = -1.173e11
energy_decrease_median = -0.0141
stop_task_fraction = 0.0
stop_decision_count = 0
filtered_action_fraction_mean = 0.0163
```

和上一版 tiny smoke checkpoint 相比，mean/median R2 基本没有改善：

```text
tiny smoke 87-task mean R2 ~= 0.7233
旧 frequency-target 正常配置 mean R2 ~= 0.7229
```

这说明 frequency posterior 的主要瓶颈不是单纯训练步数太少，而是 target 本身
缺少对当前 support 的 dense counterfactual ranking。

---

## 4. 指标解读

### 4.1 R2 和 solution rate

`mean R2 ~= 0.723`，`median R2 ~= 0.871` 表示模型经常能构造出有一定解释力的列空间，但离精确解很远。

`solution_rate = 0.0` 是最关键问题。旧 one-step action 语义 reward 方法在 archived 87-task 表上曾达到明显更高的 solution rate。因此当前方法不能只用“理论更干净”解释，必须继续修训练信号和推理策略。

### 4.2 NMSE

`mean NMSE ~= 0.277`，`median NMSE ~= 0.129` 与 R2 一致：不少任务有中等拟合质量，但高精度任务数量不足。

### 4.3 complexity

`mean complexity ~= 38.85`，`median complexity ~= 39`。复杂度不算失控，但因为 STOP 从未被选中，所有任务都跑满 6 步，表达式没有真正学会早停。

### 4.4 steps 和 STOP

```text
mean steps = 6.0
stop_task_fraction = 0.0
stop_decision_count = 0
```

这说明 STOP 虽然进入 support，但当前模型没有学会选择 STOP。可能原因：

```text
STOP 在大 support 中初始概率很低；
训练轨迹里早停样本覆盖不足；
当前 path weight 不显式奖励“已经足够好就停止”；
STOP feature 太弱，只是一个虚拟 action row；
推理时从 uniform p0 起步，STOP 没有先验优势。
```

STOP 需要进一步加强，但不能用局部 residual reward 伪造目标。更合理的是增加 STOP 的 on-policy 覆盖和显式长度/复杂度在 terminal reward 中的作用。

### 4.5 filtered_action_fraction

```text
filtered_action_fraction_mean ~= 0.0163
```

健康过滤比例很低，说明当前过滤没有大规模删掉候选，也没有解释性能差。它主要防止极端无效 action。

### 4.6 energy_decrease

```text
energy_decrease_mean = -1.173e11
energy_decrease_median = -0.0141
```

median 接近 0，但 mean 巨大负数，说明少数任务仍有严重数值异常或 energy 爆炸。当前健康过滤只过滤单步候选语义和单步 energy growth，但没有完全防住多步组合后的病态列空间。

这应作为数值稳定性问题处理：

```text
更强的候选健康过滤；
最终 active column 健康选择；
readout 前剔除病态列；
报告 outlier task；
限制 exp/cube 的重复组合。
```

不要把 local residual reward 加回主 target。

---

## 5. 为什么旧 frequency target 理论更干净但结果没好

当前理论修正解决的是“目标定义脏”的问题：

```text
不再把完整轨迹 reward 硬压成 block reward；
不再用 H×A/zeta 近似作为主线；
不再混入 local residual reward；
用 q*(a|s) 做 path-posterior 条件边缘。
```

但它没有自动解决三个工程/统计问题。

### 5.1 path-posterior 估计方差仍然高

旧 frequency target 的 `q*(a|s)` 只来自采样到的轨迹访问。当时每轮：

```text
32 tasks x 24 trajectories
```

仍然偏少。很多 state/action 的访问次数很低，`q*` 估计噪声大。

### 5.2 behavior policy 初期太弱

如果 behavior model 早期采不到高质量轨迹，那么 risk-weighted path measure 只能在低质量样本中重加权。它比旧 dense one-step oracle 更依赖采样覆盖。

### 5.3 推理和训练分布仍不完全一致

旧版本训练时每个 state 使用 capped support：

```text
max_support_size = 64 + STOP
```

旧版本推理时默认使用完整健康 support + STOP。当前版本已经让推理默认读取
checkpoint 的 `max_support_size`，按训练顺序执行 deterministic support cap、
health filter、STOP append。

---

## 6. 当前新增 support budget 的意义

full action support 在 `K=10` 时仍然会产生很大的 Gram 和 record：

```text
Gram size = |A_s| x |A_s|
```

不做 support budget 时，正常训练配置会产生数 GB 级别 records，甚至无法完成。当前加入：

```text
max_support_size = 32
```

这是计算 support 限制，不是 reward 近似。它的代价是：

```text
q_hat(a|s) 只在 capped local support 上定义；
未进入 support 的 action 没有本轮 target。
```

这个近似必须在论文或实验文档中诚实说明。

---

## 7. 下一步优先级

### P0：让训练和推理使用同一个 support builder

已实现：推理默认读取 checkpoint 中的 `max_support_size`，并按训练 sampler 的顺序执行：

```text
deterministic support cap
health filter
STOP append
feature/effect construction
```

这样训练和推理看到的 action simplex 一致。

### P1：增强 STOP 学习

建议先做不污染目标的改动：

```text
terminal reward 加复杂度/长度惩罚；
提高 STOP 在 behavior support 中的覆盖；
记录 STOP 出现次数、STOP q*、STOP rank；
训练/推理统一 STOP prior 或初始化 bias；
评估 STOP oracle：如果当前 active columns 已足够好，STOP 是否应成为 q* top action。
```

### P2：加入 oracle / exact / learned 三层诊断

必须拆开：

```text
sampled trajectory oracle R2
path-posterior exact teacher rollout R2
learned model rollout R2
```

否则无法判断差距来自：

```text
采样覆盖差
q* 估计差
semantic-Fisher teacher 差
模型拟合差
推理策略差
```

### P3：数值稳定性

建议新增：

```text
active column health report
readout 前健康列筛选日志
energy outlier task list
operator repetition limits for exp/cube/protected_div
final semantic max_abs / condition number diagnostics
```

这些是 validity / diagnostics，不应进入 reward 主目标。

### P4：提高 on-policy 样本效率

当前配置能跑，但还偏小。扩大前先优化：

```text
dataset streaming
state/support cache
semantic effect cache
训练进度日志
按 task 分批构建 records
```

然后再提高：

```text
num_tasks
num_trajectories
on_policy_iterations
teacher_steps
```

### P5：旧 one-step 方法作为 regression baseline

旧 one-step residual 方法不再作为主线目标，但保留为 `OneStepTarget`
sanity baseline。当前新增的 `FutureGroup-L3Target` 把旧 one-step 奖励扩展为
短 horizon group rollout 评分，用作独立实验组；它和主理论共享同一个
semantic-Fisher ODE。

---

## 8. 当前结论

当前 Semantic-Fisher Flow Matching 改动把理论对象进一步解耦：

```text
TargetSampler/reward/search -> q_hat
semantic effects -> geometry
flow matching -> lambda-time velocity
```

它删除了旧 frequency target 的主动实现，也不再把 terminal semantic projection
作为主线。当前 `FutureGroup-L3Target` 87-task 结果保存在：

```text
results/semantic_fisher_flow_87/semantic_fisher_flow_future_group_l3_87_seed0_summary.json
```

剩余风险是：

```text
采样覆盖
STOP 学习
support 一致性
数值 outlier
oracle/exact/learned 诊断缺失
```

下一轮不应回退到 local residual reward，而应先补上述诊断和一致性问题。
