# Results Directory

当前活跃结果按 `Semantic-Fisher Flow Matching` 口径整理。实验组由
`TargetSampler` 区分；旧 path-posterior frequency 和 semantic-projection smoke
结果目录已经清理。

## Active

| Path | 内容 |
|---|---|
| `semantic_fisher_flow_87/` | 当前 action-flow 两组 TargetSampler 在 legacy 87-task benchmark 上的正式结果 |
| `dataset_validation/` | 324-task benchmark manifest 校验报告 |
| `benchmark_plans/` | 外部 baseline / SFSR 命令计划 |
| `baseline_sanity/` | baseline adapter 的轻量 sanity 输出 |
| `block_risk_flow_87/`, `block_risk_flow_smoke/`, `risk_flow_smoke/` | 旧 block/H1 diagnostic 结果，仅作回溯参考 |

## Semantic-Fisher Flow Matching 87-Task Runs

| TargetSampler | Summary | Mean R2 | Median R2 | Solution rate | STOP task fraction |
|---|---|---:|---:|---:|---:|
| `OneStepTarget` | `semantic_fisher_flow_87/semantic_fisher_flow_one_step_87_seed0_summary.json` | 0.8130 | 0.9036 | 0.1494 | 0.1034 |
| `FutureGroup-L3Target` | `semantic_fisher_flow_87/semantic_fisher_flow_future_group_l3_87_seed0_summary.json` | 0.8466 | 0.9122 | 0.1149 | 0.0575 |

Commands used:

```bash
conda run -n semflow python -m semflow_sr.train.train_path_posterior_flow \
  --config configs/train/semantic_fisher_flow_87_one_step.yaml

conda run -n semflow python scripts/run_path_posterior_flow.py \
  --ckpt checkpoints/semantic_fisher_flow_one_step_87.pt \
  --legacy_87 \
  --out results/semantic_fisher_flow_87 \
  --tag semantic_fisher_flow_one_step_87_seed0 \
  --max_steps 6 \
  --device cpu

conda run -n semflow python -m semflow_sr.train.train_path_posterior_flow \
  --config configs/train/semantic_fisher_flow_87_future_group_l3.yaml

conda run -n semflow python scripts/run_path_posterior_flow.py \
  --ckpt checkpoints/semantic_fisher_flow_future_group_l3_87.pt \
  --legacy_87 \
  --out results/semantic_fisher_flow_87 \
  --tag semantic_fisher_flow_future_group_l3_87_seed0 \
  --max_steps 6 \
  --device cpu
```

## Archived

旧结果在：

```text
results/archive_legacy_20260618/
```

其中内容只能用于回溯诊断，不作为当前主线结果。
