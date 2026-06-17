# 项目架构与目录说明

当前代码按四层看最清楚：

```text
semantic state + centered projection
-> action effect extraction
-> semantic-Fisher geometry
-> learned local update operator
```

## 顶层结构

```text
SemanticFlowSR/
├── configs/
├── semflow_sr/
├── scripts/
├── tests/
├── docs/
├── checkpoints/   # generated
└── results/       # generated
```

## 关键模块

### `semflow_sr/semantics/`

- `projection.py`: centered ridge projection backend, including `residual_vector`.
- `energy.py`: one-step reward and `action_semantic_effects`.

### `semflow_sr/actions/`

- `action_space.py`: legal action enumeration.
- `action_executor.py`: symbolic / semantic action execution.
- `action_features.py`: static action features.
- `support_sampler.py`: support approximation.

### `semflow_sr/flow/`

- `natural_path.py`: old exponential Fisher path utilities; now ablation/support code.
- `semantic_fisher.py`: current mainline solver and sphere step.

### `semflow_sr/models/`

- `semantic_transformer.py`: row/register/action encoders plus main `lograte` head.
- `action_encoder.py`: action-relation mixing using `gram`.
- `velocity_model.py`: bridges raw head output to `lograte_logits`, `v_pred`, `z_dot_pred`.

### `semflow_sr/data/`

- `trace_dataset.py`: builds semantic-Fisher target records.
- `collate.py`: pads support-local tensors including `gram`, `xi`, `semantic_stats`.

### `semflow_sr/train/`

- `trainer_velocity.py`: current main trainer.
- `losses.py`: `SemanticFisherVelocityLoss` mainline; `SpherePathLoss` retained for the plain Fisher ablation.
- `build_dataset.py`: synthetic trace dataset builder.
- `train_velocity_gt.py`, `train_base_natural_flow.py`: train entry points.

### `semflow_sr/inference/` and `semflow_sr/search/`

- `iterative_policy_update.py`: semantic-Fisher and closed-form update helpers.
- `rollout_velocity.py`: actual SR rollout and diagnostics; default is `semantic_fisher_sphere`.

### `semflow_sr/targets/` and `semflow_sr/gp_distill/`

These are providers and extensions:

- one-step / rollout / search targets provide scalar scores
- GP modules expose interfaces but do not alter the main geometry

## 当前主数据流

### 训练

```text
trace step
-> B, y, support S
-> centered residual e and action effects xi
-> Gram K and semantic_stats
-> provider scores R(a)
-> normalized advantage A(a)
-> exact log-rate w_target
-> sphere tangent z_dot_target
-> model predicts w_theta
-> semantic_fisher_velocity loss
```

### 推理

```text
state
-> support S
-> centered residual / action effects / gram
-> model log-rate w_theta
-> semantic_fisher_sphere_step
-> argmax / sample action
```

## 仍保留但不是主线

- `gamma=0` semantic-Fisher no-pullback setting.
- `closed_form_policy_update` for the plain Fisher potential endpoint ablation.
- `SpherePathLoss` for plain Fisher sphere-path training comparisons.
- `endpoints/` 的旧 `p0/p1` 兼容接口，用于旧数据/脚本读写。

这些分支现在只服务于回归测试、旧 checkpoint 兼容或明确 ablation。
