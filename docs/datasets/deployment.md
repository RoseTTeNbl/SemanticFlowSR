# 数据集部署

所有命令在 `SemanticFlowSR/` 下、`semflow` 环境中执行。

## S1：局部 semantic-Fisher target

Trainer 可以在线生成；如果要单独落盘 trace dataset，主配置应该与当前算法一致：

```bash
python scripts/generate_trace_dataset.py \
  --num_tasks 2000 \
  --num_vars 1 \
  --max_depth 4 \
  --K 8 \
  --probe_size 128 \
  --target one_step_advantage \
  --max_support 128 \
  --support_mode mixed_topk_random \
  --out data/local_flow_traces/train
```

当前主链不再依赖闭式 path sample。核心是：

- centered residual backend
- `xi`
- `gram`
- `w_target`
- `zdot_target`

## S2：公式 benchmark 物化

```bash
python scripts/materialize_formula_benchmark.py \
  --suite nguyen constant livermore jin \
  --seeds 0 1 2 3 4 \
  --out data/materialized
```

## 评测主命令

```bash
python scripts/run_experiment.py \
  --ckpt checkpoints/velocity_one_step_advantage.pt \
  --suite nguyen constant livermore jin \
  --seed 0 \
  --out results/semantic_fisher \
  --tag formula_1var_seed0 \
  --integration_method semantic_fisher_sphere \
  --step_size 1.0 \
  --beta 1.0 \
  --gamma 0.1
```

## PMLB / SRBench

PMLB loader 与缓存脚本仍可用。它们只是 benchmark 输入源，不会改变 semantic-Fisher target 的构造方式。
