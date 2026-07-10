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
2. 当前速度场 `v_psi` rollout 生成 reference endpoint proposal；`G_phi` 只是可选 one-step student。
3. tilt 在具体 trace 上做：argmax / temperature sampling / graph mutation / archive，按 held-out raw+affine 指标选 elite atoms。
4. elite trace 投影为 `P_epsilon(z;theta0)`：active blocks 尖锐，inactive blocks 保持 source identity。
5. capacity resampling + active-only Fisher Hungarian assignment 保持 tracked source marginal。
6. flow `v_psi(theta_t, theta0, D, t)` 匹配稳定 Fisher-Rao bridge tangent；student 未过 gate 时 eval 回退 flow。
7. token construction、target-conditioned Stage1、semantic latent endpoint、semantic endpoint correction、online semantic mass guidance 只保留为 legacy/failed-probe evidence；当前不要作为默认实验入口。
```

当前推荐入口：

```bash
SCALE=smoke RUN_GPU=0 scripts/run_one_step_semantic_fisher_cycle_gpu.sh
SCALE=overfit RUN_GPU=0 scripts/run_one_step_semantic_fisher_cycle_gpu.sh
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
