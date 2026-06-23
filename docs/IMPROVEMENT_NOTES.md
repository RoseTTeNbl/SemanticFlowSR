# 改进记录与当前瓶颈

本文档只记录当前 CSEF 主线的下一步问题。

## 1. 当前证据

2026-06-23 完整结果：

```text
Fisher:
  R2 mean          0.937383
  solution rate    0.323529
  skeleton acc     0.029412
  complexity mean  12.9118

Euclidean:
  R2 mean          0.940868
  solution rate    0.294118
  skeleton acc     0.000000
  complexity mean  12.3235
```

训练侧：

```text
Fisher batch_loss mean      0.020235
Euclidean batch_loss mean   0.030072
Fisher calibration energy   0.126022
Euclidean calibration energy 0.239929
GT-neighborhood compile success on nonzero teacher rows: 0.8956
```

## 2. 当前判断

```text
Fisher path 在训练侧更容易拟合。
Euclidean path 在当前 34-task R2 均值上略高，但差异不显著。
结构恢复率仍很低，说明主要瓶颈不是单纯的概率路径几何选择。
```

当前问题应被拆成：

```text
1. structural denoising target 是否能把局部概率推向 clean GT decisions；
2. GT-neighborhood 扰动是否产生了有效 noisy context；
3. 隐参数网络是否学会把局部速度迁移到 benchmark 任务；
4. 推理采样是否能把学到的局部偏好组合成正确表达式；
5. sparse head 是否掩盖了结构错误。
```

## 3. 优先改进项

### P0：结构对齐诊断

需要按 suite 和 task 输出：

```text
GT canonical compile success
GT-neighborhood compiled count
per-task best structure_score
per-task skeleton match
per-task formula token accuracy
chosen expression vs GT
```

目标是区分：

```text
训练 target 没覆盖正确结构
训练覆盖了但网络没学到
网络学到局部偏好但推理采样没组合出来
```

### P1：Structural Denoising 目标质量

当前目标是：

```text
p_{i,1}=(1-epsilon) delta_{z_i^dagger}+epsilon uniform_i
```

需要优先检查：

```text
semantic_teacher_target_mode
semantic_teacher_clean_trace_match_rate
GT canonical compile success
GT-neighborhood noisy context compiled count
per-group clean target coverage
```

reward / R2 不进入结构 teacher target，只用于诊断和推理 rerank。

### P2：推理采样

当前评估使用 `eval_samples=64` 和 policy sampling。后续可检查：

```text
top-k or beam over local decisions
diversity-preserving sampling
validation rerank sensitivity
structure-prior rerank weight sensitivity
sample count scaling curve
per-suite sample budget sensitivity
```

重点是看 skeleton accuracy 是否随采样预算提升。

### P3：语义校准尺度

Euclidean 的 calibration energy 明显更高。需要记录：

```text
semantic_calibration_energy by decision group
source/head/operator group split
candidate semantic Gram rank
largest semantic eigenvalue
velocity norm before/after clipping
```

如果少数 group 主导能量，应局部调节 `semantic_calibration_gamma` 或重新构造输出向语义。

### P4：Sparse Head 影响

高 R2 和低 skeleton accuracy 同时出现，说明 sparse head 可能补偿了结构错误。需要额外报告：

```text
raw expression R2 before coefficient fit
calibration gain
head coefficient norm
nonzero head terms
best raw term R2
```

并按 solved / unsolved task 分组比较。

## 4. 下一步实验

推荐顺序：

```text
1. 生成 per-task structure diagnostics 表；
2. 对 Fisher 和 Euclidean 做 sample budget scaling；
3. 对 structural denoising 版本重新跑 Fisher/Euclidean；
4. 对 Jin suite 做定向错误分析；
5. 评估 explicit sum/head-as-structure 与 affine constant slots。
```

当前不应把 R2 均值的小差异解释为理论胜负；应优先解决 skeleton accuracy 为 0 的结构问题。
