# Results Layout

当前 SemanticFlowSR 主线结果集中在：

## Current Mainline

```text
results/clean_benchmark_20260701/ablations/complete_expression_semantic_fm_20260707/runs/
```

当前可见主线 run：

```text
one_step_semantic_fisher_cycle_cpu_two_iter_20260710/
```

该 run 是 2026-07-10 One-Step Semantic Fisher Cycle 的两轮 CPU 小规模闭环验证。
它证明第二轮会从更新后的 proposer 重新收集 endpoint，并记录 flow/proposer loss、
coupling、proposal 和样例输出；它不是 SR 收敛证明。

失败探针和旧路线结果不要放在当前 run 列表里。若需要保留诊断证据，放在：

```text
results/clean_benchmark_20260701/ablations/complete_expression_semantic_fm_20260707/runs/_legacy_failed_20260710/
```

其中包括 semantic latent endpoint、theta0/register 和 target-conditioned Stage1
相关失败/历史探针。

## Archive

```text
results/_archive/
```

归档内容只作为失败诊断证据，不再作为当前主线指标。
