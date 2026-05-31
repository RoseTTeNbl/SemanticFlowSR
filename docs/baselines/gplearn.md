# gplearn（遗传编程基线）

经典遗传编程符号回归，纯 Python，轻量易装。

## 环境

```bash
conda create -n gplearn python=3.11
conda activate gplearn
pip install gplearn scikit-learn pandas
pip install -e .           # 在 SemanticFlowSR/ 下
```

## 运行

```bash
python scripts/run_gplearn_baseline.py --data data/materialized/nguyen --out results/gplearn
```

- `--data`：物化套件目录；`--seed`：种子（默认 0）。

产出 `results/gplearn/gplearn_seed{k}.json`。适配器：`semflow_sr/eval/baselines.run_gplearn`。
