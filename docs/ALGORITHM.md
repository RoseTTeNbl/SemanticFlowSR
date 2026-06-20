# 算法文档：Semantic-Fisher Flow Matching

当前主线是：

```text
Semantic-Fisher Flow Matching
```

核心拆分：

```text
TargetSampler 只负责产生目标概率形 q_hat；
semantic effects xi(a) 只负责定义 semantic-Fisher geometry；
flow matching 只学习从 p_init 到 q_hat 的 lambda-time 速度场。
```

---

## 1. State, Support, Init

局部决策单元仍然是一个可执行 action：

```text
omega = action
```

每个 state `s` 构造同一个 deterministic support：

```text
A_s = legal one-step actions + STOP
```

工程上允许 `max_support_size` 作为计算近似。当前实现先 deterministic cap，再做
数值健康过滤，然后追加 STOP；训练和推理共享这个顺序。

起点概率形：

```text
p_init(a|s) = PriorBuilder(s, A_s)
```

第一版是：

```text
real legal actions: uniform logits
STOP: stop_bias_base + stop_bias_slope * construction_step
```

---

## 2. TargetSampler 实验设置

统一接口由 `semflow_sr/path_posterior/target_sampler.py` 实现：

```text
state, action_ids, p_init, x, y, rng -> TargetShape
```

返回：

```text
q_hat          # endpoint probability shape
target_scores  # reward/search scores
target_counts  # rollout/eval counts
diagnostics
```

TargetSampler 不计算 `xi`、不构造 Gram、不解 semantic-Fisher ODE。

当前实现的实验组如下。它们共享同一个 Semantic-Fisher Flow Matching
teacher/训练/推理流程，只改变 `q_hat` 的构造方式。

| 实验设置 | Config `target_mode` | TargetSampler | 目标分布构造 |
|---|---|---|---|
| `OneStepTarget` | `one_step` | `OneStepTargetSampler` | 用旧 dense one-step residual gain `E(B_s)-E(B_s^a)` 给每个 action 打分，再 rank-softmax 得到 `q_hat`。这是 sanity/regression baseline。 |
| `FutureGroup-L3Target` | `future_group_l3` | `FutureGroupTargetSampler` | 先执行候选 action，再采样长度 `L=3` 的短 continuation，用 top-k mean 聚合 rollout reward，再 rank-softmax 得到 `q_hat`。这是当前正式实验组。 |
| `CachedTrajectoryFitnessTarget` | `cached_trajectory_fitness` | `CachedTrajectoryFitnessTargetSampler` | 读取 cached trajectory fitness 记录，把进入当前 support 的首个 action 按 fitness 权重累积成经验概率形 `q_hat`。 |
| `GPCandidateFitnessTarget` | `gp_candidate_fitness` | `GPCandidateFitnessTargetSampler` | 读取训练好的 GP population/trajectory records，用可计算的 `gp_logprob` 或 event log likelihood 与 fitness 形成采样权重，抽样得到 simplex 上的一个 `q_hat` 点。 |
| `ImportanceSamplingTarget` | `importance_sampling` | `ShapeSamplingTargetSampler` | 以 one-step score 诱导目标密度、以 `p_init` 为 proposal，做 self-normalized importance sampling 得到 `q_hat`。 |
| `MCMCShapeTarget` | `mcmc_shape` | `ShapeSamplingTargetSampler` | 在 action support 的目标密度上运行 Metropolis 链，经验访问分布平滑后作为 `q_hat`。 |

`PathPosterior-Frequency` 已从这条实现中删除。Terminal semantic projection 不再是主线。

注意：TargetSampler 采样的是 endpoint probability shape `q_hat`，不是在训练时
替算法 commit 一个 action。action 的选择只发生在推理阶段的 flow integration 之后。

---

## 3. FutureGroup-L3Target

对每个 action `a in A_s`：

```text
execute a -> s^a
sample M short continuations of length L=3
compute G_{a,m} = E(B_s) - E(B_terminal) - lambda_c * complexity
S_s(a) = top-k-mean({G_{a,m}})
```

STOP 作为普通 support action：

```text
S_s(STOP) = 0
xi_s(STOP) = 0
```

目标概率形用 rank-softmax 构造：

```text
q_hat_s = rank_softmax(S_s)
q_hat_s = normalize(q_hat_s + eps * p_init)
```

使用 rank-softmax 是为了避免不同任务、不同 state 的 reward scale 污染目标。

---

## 4. Semantic-Fisher Endpoint ODE

给定：

```text
A_s
p_init
q_hat_s
xi_s(a)
K_s(a,a') = <xi_s(a), xi_s(a')>
```

先平滑 endpoint：

```text
q_eps = (1 - eps_q) q_hat_s + eps_q p_init
```

在每个 lambda-time policy `p_lambda` 上重新计算：

```text
r_lambda = log q_eps - log p_lambda
```

teacher 解：

```text
(I + gamma K_s P_lambda) w_lambda
    = beta (r_lambda + nu_lambda 1)

p_lambda^T w_lambda = 0
```

速度：

```text
dot p_lambda = p_lambda * w_lambda
dot z_lambda = 0.5 * sqrt(p_lambda) * w_lambda
```

训练目标：

```text
loss = || dot z_theta - dot z_lambda ||^2
```

这不是 endpoint classifier；lambda-time ODE 仍然是训练主对象。

---

## 5. 训练流程

入口：

```text
semflow_sr/train/train_path_posterior_flow.py
```

流程：

```text
for iteration in on_policy_iterations:
    clone current model as behavior model
    sample root behavior trajectories only to collect prefix states
    for selected prefix state:
        build support A_s
        build deterministic p_init
        TargetSampler builds q_hat
        compute xi and Gram
        integrate semantic-Fisher endpoint ODE
        write flow-matching records along lambda-time
    train model on records
```

CPU 限制由 config 的 `runtime` 字段控制：

```yaml
runtime:
  torch_num_threads: 4
  torch_num_interop_threads: 1
```

---

## 6. 推理流程

入口：

```text
scripts/run_path_posterior_flow.py
```

流程：

```text
state = initial registers
for step in max_steps:
    build same deterministic support A_s
    build p_init with the same STOP bias rule
    compute xi and Gram
    model predicts lambda-time log-rate
    integrate one semantic-Fisher sphere step
    commit argmax action or STOP
```

推理不调用 TargetSampler；TargetSampler 只用于训练 endpoint construction。

---

## 7. 主要代码位置

| 功能 | 文件 |
|---|---|
| STOP、support、health filter | `semflow_sr/path_posterior/action_support.py` |
| prefix trajectory records | `semflow_sr/path_posterior/target.py` |
| TargetSampler / p_init | `semflow_sr/path_posterior/target_sampler.py` |
| dataset 和 teacher records | `semflow_sr/path_posterior/dataset.py` |
| endpoint ODE | `semflow_sr/flow/semantic_fisher.py` |
| 训练入口 | `semflow_sr/train/train_path_posterior_flow.py` |
| 推理入口 | `scripts/run_path_posterior_flow.py` |

---

## 8. 快速命令

```bash
conda run -n semflow pytest -q tests/test_path_posterior_flow.py

conda run -n semflow python -m semflow_sr.train.train_path_posterior_flow \
  --config configs/train/semantic_fisher_flow_87_future_group_l3_smoke.yaml

conda run -n semflow python scripts/run_path_posterior_flow.py \
  --ckpt checkpoints/semantic_fisher_flow_future_group_l3_87_smoke.pt \
  --legacy_87 \
  --limit_tasks 5 \
  --out results/semantic_fisher_flow_smoke \
  --tag semantic_fisher_flow_future_group_l3_smoke_seed0 \
  --max_steps 6 \
  --device cpu
```
