# PySR（推荐工程基线）

基于 Julia 后端的高性能符号回归，工程上成熟、易复现。

## 环境

```bash
conda create -n pysr python=3.11
conda activate pysr
pip install pysr           # 首次运行会自动安装 Julia 后端
pip install -e .           # 在 SemanticFlowSR/ 下，以便 import semflow_sr.eval.baselines
```

## 运行

```bash
python scripts/run_pysr_baseline.py --data data/materialized/nguyen --out results/pysr
```

- `--data`：物化后的某个套件目录（含 `<name>/seed_k_train.csv`）。
- `--seed`：使用哪个种子的 CSV（默认 0）。
- `--niterations`：PySR 迭代数（默认 100）。

产出 `results/pysr/pysr_seed{k}.json`，每个数据集记录拟合表达式与 R²。

适配器：`semflow_sr/eval/baselines.run_pysr`（PySR 依赖惰性导入，不污染核心环境）。
