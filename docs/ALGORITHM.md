# 算法实现

本文档描述语义条件局部速度流的实现方式。理论符号对照见 [THEORY_MAPPING.md](THEORY_MAPPING.md)。

## 流水线（一次训练步）

```
任务 → 探针 (x,y) → 寄存器状态 → B = eval(state, x)            # semantics/
   → 合法动作支撑 A(state)                                     # actions/action_space
   → 逐候选 Bᵃ = T_a(B)                                       # actions/action_executor
   → E_{B,y}(a)                                              # semantics/energy
   → w(a) = exp(-η/2·E)                                      # geometry/weights
   → p0（均匀/文法）、p1（GT / 语义先知）                       # endpoints/
   → λ ~ U(0,1)
   → (p_λ, ṗ_λ) 语义 Fisher slerp                            # geometry/slerp_path
   → v_θ(p_λ, B, y, λ)                                       # models/semantic_transformer
   → 损失 = ‖v_θ − ṗ_λ‖²（严格速度匹配）                        # train/losses
```

## 核心对象

### 语义矩阵与投影（`semantics/`）
在大小为 m 的探针上求值 K 个寄存器表达式得到 `B ∈ R^{m×K}`。岭投影
`Π_{B,ρ}=B(BᵀB+ρI)⁻¹Bᵀ`。我们**从不构造 m×m 投影**，全部经由 K×K Gram 矩阵：
- `project_y`：`B(G+ρI)⁻¹Bᵀy`
- `effective_rank`：`Tr((G+ρI)⁻¹G)`
- `projection_distance`：`‖Π1−Π2‖²_F = Tr(Π1²) + Tr(Π2²) − 2 Tr(Π1Π2)`。岭投影**非幂等**，
  故对角项用 `Tr(Πᵢ²)=Tr(MGMG)`（不是 `Tr(Πᵢ)`）；交叉项 `Tr(M1·B1ᵀB2·M2·B2ᵀB1)`，
  `Mi=(Gi+ρI)⁻¹`。

所有量都在前置动作维 `[A,m,K]` 上批量计算。

### 动作能量（`semantics/energy.py`）
`E = ½‖y−Π_a y‖² + λ_r·r_eff(Bᵃ) + λ_m·‖Π_a−Π_B‖²_F + λ_op·C_op(a)`，在候选支撑上向量化。
这是速度流**唯一**的语义条件信号。

### 语义 Fisher 图（`geometry/semantic_chart.py`）
`S(p)=w√p/‖w√p‖`（单位球面上的点）；逆 `p∝z²/w²`。当 `w=1` 时退化为普通平方根/Hellinger 图。

### slerp 路径 + 闭式速度（`geometry/slerp_path.py`）
`z_λ = sin((1-λ)θ)/sinθ·z0 + sin(λθ)/sinθ·z1`，`ż_λ` 解析；
`p_λ(a) ∝ z_λ(a)²/w(a)²`；
`ṗ_λ(a) = p_λ(a)[2ż_λ(a)/z_λ(a) − Σ_b p_λ(b)·2ż_λ(b)/z_λ(b)]`。
边界处理：小 θ → 归一化线性插值；`z_λ` 远离 0 截断；端点平滑保证正支撑；`ṗ_λ` 重新中心化以强制 `Σ ṗ = 0`。

## 动作 / 寄存器模型（`actions/`、`registers/`）
K 个定长寄存器的程序。动作 `a=(op, r1, r2, write)` 用 `op(B[:,r1], B[:,r2])` 覆盖写入 `write` 列。
编码是 `(op, r1, r2, write)` 的双射混合进制索引；一元算子规范化 `r2=0`。`valid_mask` 要求读寄存器活跃。
语义执行（数值、向量化）与符号执行（rollout 用）结果完全一致（有测试）。

## 速度模型（`models/`）
Semantic Transformer：
- **RowEncoder**：对探针行 `concat(x, y, B[i,:], r_i)` 的行置换不变 Transformer。
- **RegisterEncoder**：逐列统计量（mean/std/norm/min/max/corr_y/corr_res/active），交叉注意到行 token。
- **ActionEncoder**：动作特征 + `(energy, weight, p_λ)`，交叉注意到寄存器 token 与 λ 条件全局上下文。
- **VelocityHead**：每动作一个标量，在合法支撑上做掩码切空间投影 `v = raw − mean(raw)`，使 `Σ v = 0`。

## 推理 / rollout（`search/rollout_velocity.py`）
从 `p0` 出发，沿 λ 网格积分 `v_θ`（单纯形上 Euler 步 + 重归一化），选 argmax/采样动作，符号执行，
重复直到残差能量 ≤ ε 或达到最大步数。`rollout_random` 提供诊断测试用的随机策略基线。

## 损失（`train/losses.py`）
合法支撑上的 `mse(v_pred, ṗ_λ)`，可选度量加权项 `g_{B,y,p}(v−ṗ)`（理论 §1.8）。**不使用**端点 KL。

## 遵守的不可妥协项
速度匹配（非分类）；无端点 KL 目标；无 GP；无图测度；流只在动作单纯形上；语义加权 Fisher 图；
闭式 `p_λ`/`ṗ_λ`；显式探针批语义；抽象端点（均匀/文法 `p0`，GT/语义先知 `p1`）；无 STOP 动作（能量阈值停机）。
