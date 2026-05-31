# 数据集描述

四层数据策略（参见 `docs/prompts/数据和实验建议.md`）。

## S1 层 — 局部速度流轨迹数据集（核心，必做）

最重要的数据集：训练速度场 `(B, y, p0, p1, λ) → ṗ_λ`。

由 `scripts/generate_trace_dataset.py` 生成。每个保存的样本是一条轨迹的一步：

```python
{ "x", "y", "B", "action_ids", "action_feats", "energies", "weights",
  "p0", "p1", "lambda", "p_lambda", "dp_dlambda", "gt_action_pos" }
```

其中 `p_lambda` / `dp_dlambda` 由语义 Fisher slerp（`geometry/slerp_path`）闭式给出。
生成链路：随机表达式树 → 编译为寄存器轨迹（`registers/compiler`）→ 逐步速度样本
（`data/trace_dataset`）。

配置：`configs/data/synthetic.yaml`。建议分组：poly_small、trig_small、mixed_unary、
multi_var。该层用于验证：语义图稳定性、`ṗ_λ` 的可学习性、路径速度误差、rollout 能量下降。

## S2 层 — 标准 SR 公式恢复（Nguyen / Constant / Livermore / Jin）

公式库在 `configs/data/formula_benchmarks/*.yaml`。物化为按种子划分的 train/test CSV
（`SRTask` 格式）。

输出：`data/materialized/<suite>/<name>/seed_{k}_{train,test}.csv` + `metadata.json`。
列约定：`[变量列..., target]`。加载器：`data/benchmark_loader.materialize_formula`。

各套件规模：Nguyen 12 条、Constant 8 条、Livermore 8 条、Jin 6 条。

## A 层 — SRBench / PMLB（第二阶段）

PMLB 数据本体藏在 Git LFS 之后。**当前状态**：上游 `EpistasisLab/pmlb` 的 LFS 预算已耗尽，
`git lfs pull` 失败。用户已**手动补齐** `external/pmlb/datasets/` 下的数据：

- **Feynman**：119 个目录，其中 **116 个含真实 `.tsv.gz` 数据**（缺
  `feynman_I_12_11`、`feynman_I_9_18`、`feynman_II_34_2a`）。
- **Strogatz**：14 个目录，仅含 `metadata.yaml`，**无数据文件**（loader 会跳过）。

每个数据集目录结构：`datasets/<name>/<name>.tsv.gz`（+ `metadata.yaml`、`summary_stats.tsv`）。
TSV 列为 `[特征列..., target]`。加载器：`data/benchmark_loader.PMLBLoader`。

`external/srbench` 克隆用于完整 SRBench 协议（known/black-box 划分、符号求解率），在 A 阶段使用。

## B 层 — SRSD / SRSD-dummy、NeSymReS 预训练（后续）

首个里程碑未部署。SRSD 测试无关变量鲁棒性（有效秩 / 投影）；NeSymReS 提供大规模公式生成器，
`registers/compiler` 已支持把完整表达式转成寄存器轨迹用于预训练。

## 目录布局

```
data/                                   # 生成物（被 .gitignore 忽略）
  local_flow_traces/v0/traces.pt
  materialized/{nguyen,constant,livermore,jin}/<name>/seed_k_{train,test}.csv
  pmlb/feynman/<name>/{train,test}.csv

external/pmlb/datasets/                 # 用户手动补齐的 PMLB 源数据
  feynman_<...>/feynman_<...>.tsv.gz
  strogatz_<...>/                       # 仅元数据
```
