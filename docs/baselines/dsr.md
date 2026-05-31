# DSR / DSO（深度符号优化）

Deep Symbolic Regression / Optimization。强化学习 + RNN 生成表达式。

> DSR/DSO 锁定较旧依赖，**必须**用独立环境，且 Python 3.10。

## 环境

```bash
git clone https://github.com/dso-org/deep-symbolic-optimization.git external/dso
conda create -n dso python=3.10
conda activate dso
cd external/dso && pip install -e ./dso
```

## 运行

```bash
python scripts/run_dsr_baseline.py --csv data/materialized/nguyen/Nguyen-1/seed_0_train.csv
```

`run_dsr_baseline.py` 会写一个指向该 CSV 的 DSR 配置并调用 `python -m dso.run`。
若当前环境未安装 dso，脚本会提示按本文创建 `dso` 环境，而非报错退出。

- `--csv`：单个训练 CSV（列为 `[特征..., target]`）。
- `--n_samples`：DSR 采样预算（默认 200000）。
