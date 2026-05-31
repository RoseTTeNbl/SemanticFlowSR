# DiffuSR / DDSR / TPSR（参考基线）

扩散式与 Transformer 式符号回归。**撰写时无稳定官方代码**，作为论文对比中的**引用结果**处理，
本仓库的可复现性不依赖它们。

`external/` 下克隆了以下参考仓库，仅供阅读：

| 目录 | 说明 |
|---|---|
| `external/TPSR` | Transformer Planning for SR（参考） |
| `external/Symbolic_Regression_With_Diffusion_Models` | 扩散式 SR（参考） |

如需复现，请按对应仓库各自的 README 在独立环境中运行，并将其报告的 R²/复杂度填入对比表。
本仓库不提供它们的适配脚本。
