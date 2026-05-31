# 数据集文档

本目录是 SemanticFlowSR 数据集的专属文档，涵盖**描述**、**部署**与**适配**三部分。

- [overview.md](overview.md) — 四层数据策略与各数据集描述、目录规范
- [deployment.md](deployment.md) — 各数据集如何获取、生成、缓存到本地
- [adaptation.md](adaptation.md) — 数据如何接入代码（loader / `SRTask` 规范 / 列约定）

总体上数据分四层：

| 层 | 数据 | 用途 | 状态 |
|---|---|---|---|
| S1 | 局部速度流轨迹 | 训练速度场 `v_θ ≈ ṗ_λ`（**核心**） | ✅ 已就绪（本地生成） |
| S2 | Nguyen/Constant/Livermore/Jin | 标准 SR 公式恢复 | ✅ 已就绪（本地物化） |
| A | SRBench / PMLB（Feynman 等） | 真实/黑盒回归基准 | ✅ Feynman 已就绪（用户手动补齐数据） |
| B | SRSD / NeSymReS 预训练 | 鲁棒性 / 大规模预训练 | ⏳ 后续里程碑 |
