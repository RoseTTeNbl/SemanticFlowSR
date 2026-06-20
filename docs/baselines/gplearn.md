# gplearn

`gplearn` 有两个用途，必须分开记录：

- 主 `semflow` 环境中的安装：用于 GP-assisted SFSR 工具和 smoke。
- 外部 baseline：作为独立方法写入 `results/external_baselines`。

## 独立环境

```bash
conda create -n gplearn python=3.11
conda activate gplearn
pip install gplearn scikit-learn pandas
pip install -e .
```

## Manifest 运行

```bash
conda run -n gplearn python scripts/run_gplearn_baseline.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --suite nguyen constant livermore jin \
  --root data/benchmark_suites \
  --generations 20 \
  --population_size 1000 \
  --out results/external_baselines \
  --tag gplearn_formula_dev
```

也可以通过矩阵入口生成命令：

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --method gplearn \
  --suite_group formula_dev
```
