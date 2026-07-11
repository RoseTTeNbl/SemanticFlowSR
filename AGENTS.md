# SemanticFlowSR agent handoff

Read this file first when working in this repository.

## 仓库根目录

```text
/home/ywj/wyh/SFSR/SemanticFlowSR
```

除非用户另有说明，使用 `semflow` conda 环境。

```bash
conda activate semflow
```

## 文档边界

完整表达式 One-Step Semantic Fisher Cycle 主线算法和架构写在：

```text
docs/ALGORITHM_COMPLETE_EXPRESSION_SEMANTIC_FM.md
docs/ARCHITECTURE_COMPLETE_EXPRESSION_SEMANTIC_FM.md
docs/MATH.md
```

诊断验证记录写在：

```text
docs/STRUCTURAL_CLOSURE.md
```

不要把 fixed-pool、GT/proxy、failure taxonomy、128-task matrix 或 distillation
诊断写入主线算法/架构文档。

只有诊断工作改变采样、训练目标或构造图行为时，才更新主线算法/架构文档。

## 训练状态

清理期间已明确停止训练。除非用户要求，不要重启 128-task、dynamic-pool、
semantic FM、structural-closure 或其他长 GPU 任务。

当前默认入口：

```text
scripts/train_complete_expression_semantic_fm.py
scripts/run_semantic_flow_gpu.sh
scripts/run_bootstrap_gates_gpu.sh
```

`target_conditioned_reference`、semantic latent endpoint、semantic endpoint correction
和 token semantic pushforward 都是历史/失败探针，不要作为默认入口恢复。

## 主要文件

```text
scripts/train_complete_expression_semantic_fm.py
semflow_sr/one_step_fisher.py
semflow_sr/latent_endpoint.py
semflow_sr/edge_flow/template.py
semflow_sr/edge_flow/path_compiler.py
```

## 编辑规则

- Prefer `rg` for searches.
- Do not modify `/home/ywj/wyh/SFSR/paper` unless the user explicitly asks.
- Keep diagnostic results in `docs/STRUCTURAL_CLOSURE.md`.
- Do not run full training unless the user explicitly asks.
- Future result and reflection documents should be written in Chinese.

重要！每次算法上的大修改需要同步到git；每次跑实验要及时检查和清理过期的log和result目录（旧的实验结果）避免过多堆叠的冗余目录
