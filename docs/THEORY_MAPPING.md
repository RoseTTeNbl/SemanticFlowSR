# 理论 → 代码 映射

把 `docs/prompts/主线理论.md`（§1.x）中的每个对象映射到其实现。

| 理论 | 符号 | 代码 |
|---|---|---|
| §1.1 语义矩阵 | `B=[b_1..b_K]∈R^{n×K}` | `registers/executor.evaluate_register_state`、`semantics/semantic_matrix` |
| §1.1 投影 | `Π_B = P_{V(B)}` | `semantics/projection.ProjectionBackend`（默认岭回归，可做 hard 消融） |
| §1.1 残差 | `r_B=(I−Π_B)y` | `ProjectionBackend.residual_energy`（½‖r‖²） |
| §1.2 张成距离 | `d²=‖Π_U−Π_V‖²_F` | `ProjectionBackend.projection_distance`（Gram-trace 形式） |
| §1.3 增益 | `G_{B,y}(a)=½yᵀ(Π_a−Π_B)y` | 经残差能量隐式使用 |
| §1.4 有效秩 | `r_eff=Tr(Π_{B,ρ})` | `ProjectionBackend.effective_rank`、`semantics/rank` |
| §1.4 动作能量 | `E_{B,y}(a)` | `semantics/energy.ActionEnergy.compute` |
| §1.5 权重 | `w=exp(−η/2·E)` | `geometry/weights.semantic_weights` |
| §1.5 图 | `S_{B,y}(p)` | `geometry/semantic_chart.semantic_chart` |
| §1.6 逆 | `p∝z²/w²` | `geometry/semantic_chart.inverse_semantic_chart` |
| §1.7 距离 | 加权 Hellinger | `geometry/distances.semantic_fisher_distance` |
| §1.8 局部度量 | `g_{B,y,p}` | `geometry/velocities.semantic_metric_norm_sq` |
| §1.9 slerp 路径 | `z_λ`、`p_λ` | `geometry/slerp_path.SemanticFisherSlerpPath.sample` |
| §1.10 速度 | `ṗ_λ` 闭式 | `slerp_path` 的 `dp_dlambda`（经 `2ż/z` 对数导数） |
| §1.11 曲率 | 加权诠释 | （概念性——由图中的 `w` 实现） |

## 速度推导（§1.10）的实现

记 `q_λ(a)=z_λ(a)²`，`p_λ(a)=(q_λ(a)/w(a)²)/C_λ`。则
`ṗ_λ(a)=p_λ(a)[q̇_λ/q_λ − Σ_b p_λ(b)·q̇_λ(b)/q_λ(b)]`，其中 `q̇/q = 2ż/z`，`ż_λ` 来自 slerp 闭式。
代码计算 `ratio = 2·dz/z`（z 远离 0 截断），`mean_ratio = Σ p_λ·ratio`，
`dp = p_λ·(ratio − mean_ratio)`，再重新中心化使 `Σ=0`。

数值验证：`tests/test_slerp_velocity.py::test_analytic_velocity_matches_finite_difference`。

## 首个里程碑刻意排除项
- `GP解释.md` / `函数空间推广.md` **不使用**（仅论文附录）。
- 无 GP 先验/目标、无图测度、无全表达式空间流、无 STOP 动作、无端点 KL 训练目标。
  这些与提示词的不可妥协规则一致。
