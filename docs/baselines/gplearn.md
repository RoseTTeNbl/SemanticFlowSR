# gplearn

`gplearn` 作为独立符号回归方法运行，并按统一 manifest 写出逐任务 JSON 结果。

## 独立环境

```bash
conda create -n gplearn python=3.11
conda activate gplearn
pip install gplearn scikit-learn pandas
pip install -e .
```

## manifest 运行

```bash
conda run -n gplearn python scripts/run_gplearn_baseline.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --suite nguyen constant livermore jin \
  --root data/benchmark_suites \
  --generations 20 \
  --population_size 1000 \
  --out results/baselines_current \
  --tag gplearn_formula_dev
```

也可以通过矩阵入口生成命令：

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --method gplearn \
  --suite_group formula_dev
```
