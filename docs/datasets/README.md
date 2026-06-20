# 数据集文档

本目录是 SemanticFlowSR 数据集的专属文档，涵盖**描述**、**部署**与**适配**三部分。

- [overview.md](overview.md) — 四层数据策略与各数据集描述、目录规范
- [deployment.md](deployment.md) — 各数据集如何获取、生成、缓存到本地
- [adaptation.md](adaptation.md) — 数据如何接入代码（loader / `SRTask` 规范 / 列约定）

当前 benchmark 分层：

| 层 | 数据 | 用途 | 状态 |
|---|---|---|---|
| S1 | Semantic-Fisher RiskFlow target | 完整轨迹 reward -> trajectory advantage -> visited block credit -> `w_θ` flow matching（**核心**） | ✅ 已就绪（本地生成） |
| Dev | Nguyen/Constant/Livermore/Jin | workflow / diagnostics / 旧结果对齐 | ✅ 34 tasks |
| Main-A | SRSD-Feynman easy/medium/hard | scientific discovery / Feynman 主表 | ✅ 120 tasks |
| Main-B | PMLB regression filtered | 黑盒回归、runtime、鲁棒性 | ✅ 50 tasks |
| Appendix | SRSD-Feynman dummy | dummy variable / feature selection | ✅ 120 tasks |
| Future | LLM-SRBench / cp3-bench | 高难科学发现 / cosmology OOD | ✅ repo 已下载，adapter 后续 |

统一入口：

```text
data/benchmark_suites/benchmark_manifest.json
data/benchmark_suites/benchmark_index.csv
```

SFSR 自身评估也支持 manifest：

```bash
conda run -n semflow python scripts/validate_benchmark_manifest.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --root data/benchmark_suites \
  --out results/dataset_validation \
  --fail-on-error
```

矩阵化 SFSR 评估入口：

```bash
conda run -n semflow python scripts/run_sfsr_benchmark_matrix.py \
  --suite_group formula_dev \
  --ckpt_by_vars 1:checkpoints/d1.pt 2:checkpoints/d2.pt \
  --plan_out results/benchmark_plans/sfsr_risk_flow_commands.json
```
