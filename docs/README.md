# SemanticFlowSR 文档

当前主线已经切到 semantic-Fisher pullback flow。以下文档都按这个版本同步：

| 文档 | 内容 |
|---|---|
| [MATH.md](MATH.md) | 当前数学规格：centered residual、semantic effect、pullback metric、log-rate solver、sphere update |
| [ALGORITHM.md](ALGORITHM.md) | 当前算法链路：target 构造、训练目标、推理更新、provider 接口 |
| [THEORY_MAPPING.md](THEORY_MAPPING.md) | 核心公式到代码实现的映射 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 项目结构、模块职责、主数据流 |
| [IMPROVEMENT_NOTES.md](IMPROVEMENT_NOTES.md) | 从 plain Fisher-sphere 到 semantic-Fisher 的修正记录与当前结论 |
| [datasets/overview.md](datasets/overview.md) | 本地 target dataset 的当前字段与用途 |
| [datasets/deployment.md](datasets/deployment.md) | 数据集和 benchmark 部署命令 |
| [EVAL_87_ADAPTATION.md](EVAL_87_ADAPTATION.md) | benchmark 适配备注；非主理论文档 |
| [baselines/](baselines/README.md) | 外部基线说明 |
| [../results/future_ode_gp_comparison.md](../results/future_ode_gp_comparison.md) | future-ODE 与 GP-guided smoke 评测记录 |

运行方式见仓库根目录 [README.md](../README.md)。
