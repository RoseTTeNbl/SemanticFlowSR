# SemanticFlowSR

当前主线：完整表达式 One-Step Semantic Fisher Cycle。

主线算法文档：

```text
docs/ALGORITHM_COMPLETE_EXPRESSION_SEMANTIC_FM.md
docs/ARCHITECTURE_COMPLETE_EXPRESSION_SEMANTIC_FM.md
docs/MATH.md
```

状态和诊断文档：

```text
docs/STRUCTURAL_CLOSURE.md
docs/DIAGNOSTIC_EXPERIMENTS_COMPLETE_EXPRESSION_FLOW.md
```

当前约定：

```text
1. theta 是 register categorical block 阵列，不是裸欧氏向量。
2. 条件速度场接口固定为 `v(theta, D, t)`；`theta0` 仅作为 ODE 初值，不作为网络条件。
3. 每个 source rollout 一个 endpoint，并只解码一个完整表达式；GT 等价 trace 仅用于 bootstrap 缓存和诊断。
4. active blocks 使用 source-mass endpoint，inactive blocks 严格保持 source identity，速度由 soft reachability gate 抑制。
5. outer loop 使用 weighted-Poisson 语义势修正同一粒子的 endpoint，不使用 matching、mutation、elite selection 或 archive。
6. 新桥相对当前场的差异由轻量 residual velocity head 学习；Gate A/B/C 未通过时禁止进入 Poisson outer loop。
7. token construction、target-conditioned Stage1、semantic latent endpoint 和旧 semantic endpoint correction 只保留为失败探针证据，不作为默认入口。
```

当前推荐入口：

```bash
SCALE=smoke RUN_GPU=0 scripts/run_semantic_flow_gpu.sh
SCALE=overfit RUN_GPU=0 scripts/run_semantic_flow_gpu.sh
```

## Environment

Run commands from `SemanticFlowSR/`.

```bash
conda activate semflow
```

If the environment must be recreated:

```bash
conda create -n semflow python=3.11
conda activate semflow
pip install --index-url https://download.pytorch.org/whl/cu126 torch
pip install numpy scipy sympy pyyaml pandas scikit-learn tqdm einops pytest
pip install deap gplearn
pip install -e .
```
