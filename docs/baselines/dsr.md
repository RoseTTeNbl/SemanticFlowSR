# DSR / DSO

Deep Symbolic Regression / Optimization。建议在独立环境运行。

## 环境

```bash
git clone https://github.com/dso-org/deep-symbolic-optimization.git external/dso
conda create -n dso37 python=3.7
conda activate dso37
cd external/dso && pip install -e ./dso
cd -
pip install -e .
```

## manifest 运行

```bash
conda run -n dso37 python scripts/run_dsr_baseline.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --suite nguyen constant livermore jin \
  --root data/benchmark_suites \
  --n_samples 100000 \
  --out results/baselines_current \
  --tag dso_formula_dev
```

矩阵入口：

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --method DSO \
  --suite_group formula_dev
```
