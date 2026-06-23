# 架构说明

当前主线：

```text
Conditional Semantic Edge Flow, CSEF
```

CSEF 代码集中在：

```text
semflow_sr/edge_flow/
```

模型不保存每个构造边的固定概率表，而是用共享条件网络在每个自回归局部上下文中生成显式概率形的速度。

## 1. 训练数据流

```text
YAML config
-> RegisterOperatorTemplate
-> ConditionalEdgeFlowModel
-> GT expression compiler
-> GT-neighborhood path sampler
-> structural denoising target shape P_1
-> source probability shape P_0
-> Fisher or Euclidean p_0 -> p_t teacher velocity
-> semantic-calibrated velocity matching
-> checkpoint + train curve CSV
```

当前训练主线使用：

```text
objective: semantic_teacher
target_shape_source: structural_denoising
teacher_target_mode: structural_denoising
teacher_time_sampling: uniform
sampler_method: policy
probability_path_geometry: fisher | euclidean
semantic_calibration_gamma: 1.0
```

## 2. 评估数据流

```text
checkpoint
-> scripts/run_edge_flow.py
-> ConditionalEdgeFlowModel + ConditionalEdgeFlowSampler
-> policy samples complete register paths
-> render sparse head terms
-> fit linear coefficients
-> score R2 / structural prior / complexity / structure metrics
-> write summary, samples, expressions, grouped statistics, diagnostics
```

完整评估目录包含：

```text
*_summary.json
*_samples.jsonl
*_task_expressions.csv
*_task_expressions.md
*_statistics_by_group.csv
*_statistics_by_group.json
*_diagnostics.json
```

## 3. 主要模块

| 文件 | 责任 |
|---|---|
| `semflow_sr/edge_flow/template.py` | register/operator/head 模板元数据 |
| `semflow_sr/edge_flow/conditional.py` | CSEF 模型、采样器、局部 logits/velocity、sparse head 渲染 |
| `semflow_sr/edge_flow/semantic_teacher.py` | Fisher/Euclidean teacher velocity、structural denoising target、语义校准 loss、teacher 诊断 |
| `semflow_sr/edge_flow/gt_neighborhood.py` | GT canonical path 周边扰动，用于 noisy context 生成 |
| `semflow_sr/edge_flow/path_compiler.py` | 公式到 CSEF path 的确定性编译 |
| `semflow_sr/edge_flow/structure_posterior.py` | 结构相似度、端点权重和局部 posterior 证据 |
| `semflow_sr/edge_flow/reward.py` | 表达式执行、head 拟合、R2/NMSE/复杂度诊断 |
| `semflow_sr/edge_flow/benchmark.py` | manifest task loader、metrics、结果文件写入 |
| `semflow_sr/edge_flow/selection.py` | R2、结构先验和复杂度组成的推理 rerank score |
| `semflow_sr/edge_flow/train_edge_flow.py` | 训练 CLI 和训练曲线写入 |
| `scripts/run_edge_flow.py` | 评估 CLI |
| `scripts/archive_paper_metrics.py` | 论文表格、置信区间、显著性和图表输出 |

## 4. 配置线

Fisher 主线：

```text
configs/train/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.yaml
```

Euclidean 消融：

```text
configs/train/conditional_edge_flow_gt_sampler_teacher_path_euclidean_gpu.yaml
```

两条线共享模型、数据、structural denoising target 和语义校准设置；只改变：

```text
probability_path_geometry
```

## 5. 训练曲线关键字段

```text
device
probability_path_geometry
batch_loss
semantic_teacher_loss_mean
semantic_calibration_loss_mean
semantic_calibration_energy_mean
semantic_teacher_trace_count
semantic_teacher_target_mode
semantic_teacher_clean_trace_match_rate
gt_neighborhood_compile_success_rate
gt_neighborhood_compiled
teacher_path_endpoint_l1_mean
teacher_path_current_entropy_mean
semantic_teacher_recomputed_velocity_rate
```

这些字段用于判断：

```text
GT-neighborhood 是否提供可训练 noisy context
teacher velocity 是否在 p_t,t 上重算
语义校准是否引入过大的局部误差尺度
Fisher 与 Euclidean 两条概率路径是否有可比较差异
```
