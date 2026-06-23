# SemanticFlowSR 文档入口

本文档目录只记录当前 CSEF 主线。路径、配置名、字段名、指标名和命令参数保持代码原名；算法解释和实验结论使用中文。

## 当前主线

```text
Conditional Semantic Edge Flow, CSEF
```

核心文档：

| 文档 | 内容 |
|---|---|
| [ALGORITHM.md](ALGORITHM.md) | 当前 CSEF 训练、推理、GT target shape 和 teacher velocity 流程 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 代码结构、模块责任、训练和评估数据流 |
| [MATH.md](MATH.md) | 显式概率形、Fisher path、Euclidean 消融和语义校准数学 |
| [RESULTS.md](RESULTS.md) | 当前 Fisher / Euclidean 结果、图表和结论 |
| [IMPROVEMENT_NOTES.md](IMPROVEMENT_NOTES.md) | 当前瓶颈和下一步改进事项 |

补充文档：

| 文档 | 内容 |
|---|---|
| [datasets/README.md](datasets/README.md) | benchmark manifest、数据生成和评估数据接口 |
| [baselines/README.md](baselines/README.md) | 外部符号回归基线和论文指标输出接口 |
| [../README.md](../README.md) | 项目快速入口 |
| [../AGENTS.md](../AGENTS.md) | 会话交接信息和工程约束 |

## 当前命令入口

训练：

```bash
conda run --no-capture-output -n semflow \
  python -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.yaml
```

评估：

```bash
conda run --no-capture-output -n semflow python scripts/run_edge_flow.py \
  --ckpt checkpoints/teacher_path_geometry/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.pt \
  --out results/teacher_path_geometry_fisher_gpu \
  --tag teacher_path_geometry_fisher_gpu \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --manifest_root data/benchmark_suites \
  --manifest_suite nguyen constant livermore jin \
  --eval_samples 64 \
  --flow_steps 1 \
  --sampler_method policy \
  --head_fit_mode linear \
  --device cuda:1
```

结果指标输出：

```bash
conda run --no-capture-output -n semflow python scripts/archive_paper_metrics.py \
  --out results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623 \
  --suite nguyen constant livermore jin \
  --method CSEF-Fisher SFSR sfsr_method samples_jsonl results/teacher_path_geometry_fisher_gpu/teacher_path_geometry_fisher_gpu_samples.jsonl \
  --method CSEF-Euclidean SFSR sfsr_method samples_jsonl results/teacher_path_geometry_euclidean_gpu_20260623/teacher_path_geometry_euclidean_gpu_20260623_samples.jsonl
```

## 文档维护规则

1. 算法变化同步更新 [ALGORITHM.md](ALGORITHM.md)、[ARCHITECTURE.md](ARCHITECTURE.md) 和 [MATH.md](MATH.md)。
2. 新实验只记录当前有效结果，写入 [RESULTS.md](RESULTS.md) 并保留 artifact 路径。
3. 失败分析写入 [IMPROVEMENT_NOTES.md](IMPROVEMENT_NOTES.md)，先列证据，再列判断和下一步。
4. 新增结果目录必须能被 `scripts/archive_paper_metrics.py` 读取或被 [RESULTS.md](RESULTS.md) 明确说明。
