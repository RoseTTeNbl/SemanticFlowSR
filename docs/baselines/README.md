# 外部基线文档

本目录描述与 SemanticFlowSR 对比的外部符号回归基线。每个基线在**各自独立的 conda 环境**
中运行，以保持核心 `semflow` 环境干净。基线适配器在 `semflow_sr/eval/baselines.py`，
由 `scripts/run_*_baseline.py` 调用。

- [pysr.md](pysr.md) — PySR（推荐的工程基线）
- [gplearn.md](gplearn.md) — gplearn（遗传编程）
- [dsr.md](dsr.md) — DSR / DSO（深度符号优化）
- [diffusion_sr.md](diffusion_sr.md) — DiffuSR / DDSR / TPSR（参考，引用论文结果）

## 报告什么（不止最终 R²）

按 `数据和实验建议.md §9`，本方法优势在于*语义高效的局部构造*，应报告：

1. 局部几何正确性（速度目标稳定性 vs 普通 Fisher/线性/logit）；
2. 语义条件效应（带 vs 不带 `(B,y)` 的路径质量）；
3. 搜索效率（等评估预算下的语义能量轨迹）；
4. 最终 SR 恢复率（Nguyen/Livermore/Constant/Jin 的 R²/复杂度；SRBench 后续）。

诊断使用 `eval/metrics.energy_decrease_ratio` 与 `search/rollout_*` 函数。
