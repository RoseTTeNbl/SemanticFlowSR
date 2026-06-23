# 算法说明：Conditional Semantic Edge Flow

本文档只描述当前有效算法：

```text
Conditional Semantic Edge Flow, CSEF
```

CSEF 的核心是：在表达式构造图的显式 categorical 概率形上做 teacher velocity matching。GT 表达式先编译为 canonical 构造路径，训练时用 noisy GT context 恢复 clean GT decision；隐参数网络根据当前语义上下文预测显式概率形上的速度。

## 1. 三层对象

### 1.1 表达式构造图

构造图规定：

```text
registers
operators
source-register choices
write/update choices
sparse-head choices
```

一次完整采样路径记为：

```text
z = (z_1, ..., z_M)
```

解释器把路径变成表达式：

```text
e = Pi(z)
```

### 1.2 显式概率形

每个局部决策组 `i` 有候选集合：

```text
A_i = {1, ..., m_i}
```

局部显式概率：

```text
p_i(. | c_i) in Delta^{m_i-1}
```

完整概率形：

```text
P = {p_i}_{i in I} in M = product_i Delta^{m_i-1}
```

路径概率：

```text
q_P(z | D) = product_i p_i(z_i | c_i)
```

### 1.3 隐参数网络

隐参数网络：

```text
psi = parameters of ConditionalEdgeFlowModel
```

它不是固定概率表，而是条件速度函数：

```text
V_psi(D, prefix, B_l, c_i, p_{i,t}, t)
```

其中：

```text
D=(X,y)
B_l: 当前寄存器语义矩阵
prefix: 当前已生成结构
c_i: 局部决策上下文
p_{i,t}: 当前显式概率状态
t: flow time
```

核心分工：

```text
Flow matching 发生在 P 上。
psi 负责条件化生成 P 上的速度。
语义只进入网络输入和误差校准，不进入 Fisher-Rao teacher path。
```

## 2. Register-Root 生成流程

每层维护 `K` 个寄存器 root：

```text
R_{l,1}, ..., R_{l,K}
```

每个寄存器包含：

```text
E_{l,r}: 符号表达式树
b_{l,r}=E_{l,r}(X): 语义向量
```

每一层对可写目标寄存器执行：

```text
1. 选择是否写入目标寄存器
2. 若写入，选择 branch operator
3. 按 operator arity 选择 source registers
4. 构造 branch expression
5. 写入目标寄存器并更新寄存器语义
```

最终 sparse head 选择少量寄存器 root：

```text
f_z(x) = c_0 + sum_{s=1}^S c_s T_s(x)
```

系数通过线性拟合得到。

## 3. GT Structural Denoising Target

给定 GT 表达式：

```text
e^dagger
```

编译器生成 canonical CSEF path：

```text
z^dagger = Compile(e^dagger)
```

编译成功时，当前实现返回一条确定 path。确定性来自：

```text
parser AST
template primitives
depth schedule
sorted write target order
first matching source register
deterministic head candidate choice
```

这个保证是当前监督所需的 canonical path 唯一性；它不声称数学等价表达式在全局上唯一。

从 canonical path 采样 noisy GT-neighborhood context：

```text
tilde z ~ K_sigma(. | z^dagger)
```

noisy path 的作用是提供被扰动的局部上下文：

```text
tilde c_i = c_i(tilde z_<i, B_l(tilde z))
```

当前扰动包括：

```text
operator replacement among compatible operators
source / leaf replacement among legal registers
head term replacement among legal head candidates
canonical GT path inclusion
```

第 `i` 个局部决策的 clean target 不是扰动样本的加权边缘统计，而是 GT clean action 的 smoothed one-hot：

```text
p_{i,1}(. | tilde c_i)
  = (1 - epsilon) delta_{z_i^dagger}
    + epsilon uniform_i
```

因此训练样本是：

```text
(tilde c_i, p_{i,0}, p_{i,1}, t)
```

而不是全局统计：

```text
sum_n w_n 1[tilde z_i^(n)=a]
```

这样避免把多个不兼容结构上下文平均成 hybrid target。

reward / R2 当前只用于：

```text
GT-neighborhood diagnostics
推理 rerank
常数或 sparse head 拟合后的候选选择
```

它不主导训练时的结构 teacher target。

## 4. 源概率形

源概率形通常是合法候选上的 mask 后均匀分布：

```text
p_{i,0}(a) = 1 / |A_i|
```

也可加入小噪声，但必须保持：

```text
p_{i,0} in Delta^{m_i-1}
```

## 5. Fisher Teacher Path

对每个局部组：

```text
s_0 = sqrt(p_0)
s_1 = sqrt(p_1)
theta = arccos(<s_0, s_1>)
```

Fisher-Rao geodesic 在 square-root sphere 上是：

```text
s_t =
  sin((1-t)theta)/sin(theta) * s_0
  + sin(t theta)/sin(theta) * s_1
```

teacher velocity：

```text
dot s_t =
  -theta cos((1-t)theta)/sin(theta) * s_0
  + theta cos(t theta)/sin(theta) * s_1
```

概率状态：

```text
p_t = s_t^2
```

网络必须在同一个 `p_t,t` 上重算预测速度：

```text
dot s_psi = 0.5 * sqrt(p_t) * centered_log_rate_psi(c_i, p_t, t)
```

基础 loss：

```text
L_i = ||dot s_psi - dot s_t||^2
```

## 6. Euclidean 消融

Euclidean 消融只改变概率路径：

```text
p_t = (1-t)p_0 + t p_1
dot p_t = p_1 - p_0
dot s_t = 0.5 * dot p_t / sqrt(p_t)
```

其余设置保持一致：

```text
GT-neighborhood
structural denoising target
semantic calibration
model architecture
evaluation budget
```

这条线用于判断 Fisher-Rao 概率路径相对普通概率坐标线性路径的影响。

## 7. 语义校准

对局部候选动作计算输出向语义：

```text
Phi_i = [phi_1, ..., phi_m] in R^{n x m}
```

列标准化后：

```text
K_i = Phi_i^T Phi_i / n
```

投影到 square-root tangent：

```text
Pi_s = I - s_t s_t^T
K_tilde = Pi_s K_i Pi_s
M_i = I + gamma K_tilde
```

最终局部 loss：

```text
e_i = dot s_psi - dot s_t^*
L_i = e_i^T M_i e_i
```

因为 `M_i` 正定，唯一最优点仍是：

```text
dot s_psi = dot s_t^*
```

所以语义校准不会改变 teacher path、teacher direction 或 endpoint；它只改变不同误差方向的训练权重。

## 8. 训练流程

入口：

```bash
python -m semflow_sr.edge_flow.train_edge_flow --config <yaml>
```

每个 task 的训练流程：

```text
1. 读取 D=(X,y,e^dagger)
2. 初始化 register roots 和语义矩阵 B_0
3. 编译 GT 为 canonical path z^dagger
4. 采样 GT-neighborhood paths 作为 noisy context
5. 对每个 active local trace 找到 canonical clean action
6. 构造 p_1=(1-epsilon) delta_{z_i^dagger}+epsilon uniform_i
7. 采样源概率 p_0 和 t
8. 根据 probability_path_geometry 构造 p_t 和 teacher velocity
9. 在同一个 p_t,t 上重算模型 velocity
10. 用 semantic calibration matrix 计算 velocity matching loss
11. 反向传播更新 psi
12. 写训练曲线和 checkpoint
```

当前主配置：

```text
configs/train/conditional_edge_flow_gt_sampler_teacher_path_semantic_gpu.yaml
configs/train/conditional_edge_flow_gt_sampler_teacher_path_euclidean_gpu.yaml
```

关键字段：

```text
algorithm: conditional_semantic_edge_flow
objective: semantic_teacher
target_shape_source: structural_denoising
teacher_target_mode: structural_denoising
gt_neighborhood_size: 16
probability_path_geometry: fisher | euclidean
semantic_calibration_gamma: 1.0
teacher_time_sampling: uniform
teacher_velocity_clip: 5.0
samples_per_task: 0
head_fit_mode: linear
```

## 9. 推理流程

推理时没有 GT。流程：

```text
1. 加载 checkpoint
2. 编码任务 D=(X,y)
3. 从源概率形开始生成局部概率
4. 按 policy 采样 operator/source/write/head 决策
5. 得到完整 path z
6. 渲染 sparse head terms
7. 拟合 linear coefficients
8. 用结构先验、复杂度和数值拟合综合 rerank
9. 输出最佳表达式
```

当前评估命令使用：

```text
sampler_method: policy
flow_steps: 1
head_fit_mode: linear
```

当前选择分数：

```text
S(z) = R2(f_z)
     + lambda_prior * log q_psi(z | D) / max(C(z), 1)
     - lambda_C * C(z)
```

其中 `lambda_prior` 对应评估脚本的 `selection_eta_logprob` 参数。

## 10. 诊断量

训练侧重点字段：

```text
probability_path_geometry
semantic_teacher_loss_mean
semantic_calibration_loss_mean
semantic_calibration_energy_mean
semantic_teacher_trace_count
semantic_teacher_target_mode
semantic_teacher_clean_trace_match_rate
gt_neighborhood_compile_success_rate
gt_neighborhood_compiled
teacher_path_endpoint_l1_mean
teacher_path_current_entropy_mean
semantic_teacher_recomputed_velocity_rate
```

评估侧重点字段：

```text
r2
nmse
solution_rate
skeleton_accuracy
simplified_symbolic_equivalence_rate
operator_dependency_accuracy
formula_bleu
formula_token_accuracy
formula_edit_distance
complexity
valid_expression_fraction
unique_expression_fraction
```

## 11. 结果输出

单方法评估输出目录：

```text
*_summary.json
*_samples.jsonl
*_task_expressions.csv
*_task_expressions.md
*_statistics_by_group.csv
*_statistics_by_group.json
*_diagnostics.json
```

论文指标输出：

```bash
python scripts/archive_paper_metrics.py \
  --out results/paper_metrics/<tag> \
  --suite nguyen constant livermore jin \
  --method CSEF-Fisher SFSR sfsr_method samples_jsonl results/teacher_path_geometry_fisher_gpu/teacher_path_geometry_fisher_gpu_samples.jsonl \
  --method CSEF-Euclidean SFSR sfsr_method samples_jsonl results/teacher_path_geometry_euclidean_gpu_20260623/teacher_path_geometry_euclidean_gpu_20260623_samples.jsonl
```

当前有效指标目录：

```text
results/paper_metrics/csef_fisher_vs_euclidean_gpu_20260623
```

## 12. 当前结论

本轮完整 GPU 结果说明：

```text
Fisher 和 Euclidean 都能稳定训练并达到较高平均 R2。
结构恢复率仍很低，Fisher 在当前 evaluator 下只有少量 exact skeleton 命中。
Fisher 的 solution rate、BLEU 和 valid expression fraction 更高。
Euclidean 的平均 R2 略高，但 paired sign test 不支持显著差异。
结构对齐仍是主要瓶颈，不能只靠 teacher path geometry 解决。
```
