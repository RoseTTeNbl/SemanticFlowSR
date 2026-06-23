# PySR

PySR 是工程成熟的符号回归基线。建议独立环境运行。

## 环境

```bash
conda create -n pysr python=3.11
conda activate pysr
pip install pysr
pip install -e .
```

## manifest 运行

```bash
conda run -n pysr python scripts/run_pysr_baseline.py \
  --manifest data/benchmark_suites/benchmark_manifest.json \
  --suite srsd_feynman_easy srsd_feynman_medium srsd_feynman_hard \
  --root data/benchmark_suites \
  --niterations 100 \
  --out results/baselines_current \
  --tag pysr_srsd_main
```

矩阵入口：

```bash
conda run -n semflow python scripts/run_external_baseline_matrix.py \
  --method PySR \
  --suite_group srsd_main
```
