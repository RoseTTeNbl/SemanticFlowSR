# 项目架构与目录说明

本文档详细描述 SemanticFlowSR 的代码组织、各目录与文件职责，以及核心数据结构。
方法层面的推导见 [ALGORITHM.md](ALGORITHM.md)（实现走读）与
[THEORY_MAPPING.md](THEORY_MAPPING.md)（理论 §1.x → 代码映射）。

## 顶层结构

```
SemanticFlowSR/
├── README.md            # 快速上手 + 运行流程
├── pyproject.toml       # 包元数据与依赖
├── .gitignore
├── configs/             # 实验配置（YAML）
├── semflow_sr/          # 核心 Python 包
├── scripts/             # 命令行入口（数据生成、基线）
├── tests/               # 24 个正确性测试
├── docs/                # 文档（本目录）
├── data/                # （生成物）轨迹数据、物化 CSV、PMLB 缓存
├── checkpoints/         # （生成物）训练好的速度模型
└── external/            # 参考仓库克隆：pmlb / srbench / TPSR / 扩散 SR
```

`data/`、`checkpoints/`、`external/` 均被 `.gitignore` 忽略（属于生成物或大体积外部数据）。

## `configs/` — 配置

| 文件 | 作用 |
|---|---|
| `data/synthetic.yaml` | 速度流轨迹数据集的生成参数（变量数、深度、K、算子集、分组） |
| `data/benchmarks.yaml` | SRBench/PMLB 套件与随机种子 |
| `data/formula_benchmarks/{nguyen,constant,livermore,jin}.yaml` | 四套公式基准的表达式定义 |
| `model/semantic_transformer.yaml` | 模型超参（hidden / row_layers / heads） |
| `train/velocity_gt.yaml` | 以 GT 端点训练的配置 |
| `train/velocity_semantic_oracle.yaml` | 以语义先知端点训练的配置 |
| `eval/standard.yaml` | rollout 推理 / 读出设置 |

## `semflow_sr/` — 核心包

按依赖从底向上分层。**理论核心**在 `semantics/` 与 `geometry/`。

### `sr/` — 符号层
随机表达式的抽象语法树与算子。

| 文件 | 作用 |
|---|---|
| `ops.py` | 精简算子集；算子 id、元数（arity）、名称↔id 映射（`NAME_TO_ID`） |
| `protected_ops.py` | 保护型算子实现，保证永不产生 NaN/Inf（除零、log、sqrt 等） |
| `ast.py` | `Expr` 节点定义与 `eval_expr`（在探针上数值求值） |
| `parser.py` | 字符串公式 → `Expr`（`parse_formula`） |
| `printer.py` | `Expr` → 字符串（可选 sympy 化简） |
| `simplify.py` | 表达式化简工具 |
| `evaluator.py` | 多表达式批量求值 `evaluate_exprs` |

### `registers/` — 定长寄存器程序模型
固定 K 个寄存器，叶子（变量/常数）初始化后，每个内部算子写入一个空闲寄存器。

| 文件 | 作用 |
|---|---|
| `state.py` | `RegisterState`（活跃掩码、各寄存器表达式）、`init_register_state` |
| `executor.py` | `evaluate_register_state`：在探针 X 上求值得到语义矩阵 `B` |
| `trace.py` | `RegisterTrace` / `TraceStep`：一条 GT 动作轨迹 |
| `compiler.py` | `compile_expr`：把完整 `Expr` 后序编译成寄存器轨迹（GT 动作序列） |

> 注意：`compiler.py` 在函数内部**惰性导入** `actions.*`，以打破 registers↔actions 的包级循环导入。请勿移回模块顶层。

### `actions/` — 一步动作空间（无 STOP）
动作 `a = (op_id, read_1, read_2, write)`，覆盖写入 `write` 列。

| 文件 | 作用 |
|---|---|
| `action_space.py` | `ActionSpace` / `ActionSpec`；(op,r1,r2,w) 双射混合进制编解码；一元算子规范化 r2=0；`valid_mask` |
| `action_masks.py` | 合法性掩码（读寄存器需活跃） |
| `action_features.py` | 每个动作的特征向量（`ACTION_FEATURE_DIM`） |
| `action_executor.py` | `execute_symbolic`（符号执行，rollout 用）与 `execute_semantic`（向量化语义执行）；二者结果一致（有测试） |

### `semantics/` — **理论核心：语义矩阵 / 投影 / 能量**

| 文件 | 作用 |
|---|---|
| `semantic_matrix.py` | 语义矩阵构造辅助 |
| `probe.py` | 探针采样 |
| `projection.py` | `ProjectionBackend`：通过 K×K Gram 矩阵实现岭回归投影；`project_y` / `residual_energy` / `effective_rank` / `projection_distance`（非幂等投影用 `Tr(Π²)`） |
| `rank.py` | 有效秩 `Tr(Π_{B,ρ})` |
| `energy.py` | `ActionEnergy`：动作能量 `E_{B,y}(a) = ½‖y−Π_a y‖² + λ_r·r_eff + λ_m·‖Π_a−Π_B‖²_F + λ_op·C_op` |

### `geometry/` — **理论核心：语义 Fisher 图 + 流**

| 文件 | 作用 |
|---|---|
| `weights.py` | 语义权重 `w = exp(−η/2·E)` |
| `semantic_chart.py` | 语义图 `S(p)=w√p/‖w√p‖` 及其逆 `p∝z²/w²` |
| `distances.py` | 加权 Hellinger 距离 |
| `velocities.py` | 局部度量 `g_{B,y,p}`（度量加权速度范数，可选损失项） |
| `slerp_path.py` | `SemanticFisherSlerpPath`：语义 Fisher slerp，给出**闭式** `p_λ` 与速度 `ṗ_λ` |

### `endpoints/` — 路径端点
| 文件 | 作用 |
|---|---|
| `base.py` | 抽象基类 `PriorEndpoint` / `TargetEndpoint` |
| `prior_uniform.py` / `prior_grammar.py` | 起点 `p0`（均匀 / 文法先验） |
| `target_gt.py` / `target_semantic_oracle.py` | 终点 `p1`（GT / 语义先知） |

### `data/` — 数据集
| 文件 | 作用 |
|---|---|
| `synthetic_generator.py` | `GenConfig` + 随机表达式 → 轨迹任务 |
| `trace_dataset.py` | `VelocityTraceDataset`：把轨迹拆成逐步速度匹配样本；`build_step_records` |
| `collate.py` | `collate_velocity`：变长动作支撑集 padding + 掩码 |
| `benchmark_loader.py` | `SRTask`、公式物化 `materialize_formula`、`PMLBLoader` |

### `models/` — 速度网络（Semantic Transformer）
| 文件 | 作用 |
|---|---|
| `row_encoder.py` | 探针行编码器（行置换不变 Transformer） |
| `register_encoder.py` | 寄存器列编码器（列统计量 + 交叉注意力） |
| `action_encoder.py` | 动作编码器（动作特征 + 能量/权重/p_λ + λ 全局上下文） |
| `velocity_model.py` | `VelocityHead`：切空间投影输出，保证支撑集上 `Σv=0` |
| `semantic_transformer.py` | 装配三个编码器与速度头 |

### `train/` — 训练
| 文件 | 作用 |
|---|---|
| `losses.py` | `velocity_mse` + 可选 `metric_weighted_velocity_loss`（严格速度匹配，不用端点 KL） |
| `build_dataset.py` | 由 `GenConfig` 构建 `VelocityTraceDataset`（GT / 语义先知端点） |
| `trainer_velocity.py` | `train_velocity` + `TrainConfig` |
| `train_velocity_gt.py` | 入口：GT 端点训练 |
| `train_velocity_semantic_oracle.py` | 入口：语义先知端点训练 |

### `search/` — 推理
| 文件 | 作用 |
|---|---|
| `rollout_velocity.py` | `rollout_velocity`（沿 λ 网格积分速度场、单纯形 Euler 步、选动作、符号执行）；`rollout_random`（随机策略基线） |
| `beam.py` | `beam_search`（按残差能量的束搜索） |

### `eval/` — 指标与评估
| 文件 | 作用 |
|---|---|
| `metrics.py` | `r2_score` / `nmse` / `accuracy_rate` / `energy_decrease_ratio` |
| `evaluator.py` | `evaluate_task`：rollout → 读出最佳寄存器表达式 → 报告 R²/NMSE/复杂度 |
| `baselines.py` | 外部基线适配器（PySR / gplearn，依赖惰性导入） |

### `utils/`
`numerical.py`（数值安全工具，`normalize_simplex` 等）、`seed.py`（`set_seed`/`get_device`）、`logging.py`、`checkpoint.py`（`save_checkpoint`/`load_checkpoint`）。

## `scripts/` — 命令行入口
| 脚本 | 对应阶段 |
|---|---|
| `generate_trace_dataset.py` | 生成速度流轨迹数据集 |
| `materialize_formula_benchmark.py` | 物化公式基准为 CSV |
| `cache_pmlb_subset.py` | 缓存 PMLB 子集为 CSV |
| `run_{pysr,gplearn,dsr}_baseline.py` | 运行外部基线 |

## `tests/` — 24 个正确性测试
覆盖：动作空间编解码、保护算子、符号≡语义执行、岭投影恒等式、语义图双射、slerp 闭式速度 vs 有限差分、轨迹编译、模型形状/切空间、小数据过拟合、rollout 优于随机策略。运行：`pytest -q`。

## 核心数据流（一次训练步）

```
任务 → 探针(x,y) → 寄存器状态 → B = eval(state, x)
   → 合法动作支撑 A(state) → 逐动作 Bᵃ = T_a(B)
   → 能量 E_{B,y}(a) → 权重 w=exp(−η/2·E)
   → 起点 p0 / 终点 p1 → 采样 λ~U(0,1)
   → (p_λ, ṗ_λ) 闭式 → 模型 v_θ(p_λ,B,y,λ)
   → 损失 = ‖v_θ − ṗ_λ‖²（严格速度匹配）
```
