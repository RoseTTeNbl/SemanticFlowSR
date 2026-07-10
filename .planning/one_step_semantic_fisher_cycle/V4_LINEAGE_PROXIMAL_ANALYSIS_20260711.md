# SemanticFlowSR v4：失败判定、简化算法与可视化口径

## 1. 当前工作流程（direct velocity 与 Fisher bridge 的关系）

`direct_velocity` 指网络在推理时直接预测速度场，而不是直接预测终点：

```text
theta0
  -- RK2 integrate v_psi(theta_t, X, y, t, theta0) --> bar_theta1
  -- terminal diagnostics / local proximal correction --> theta1+
  -- analytic Fisher bridge(theta0, theta1+) --> supervised velocity target
  -- flow matching --> next v_psi
```

因此：

- 推理终点 `bar_theta1` 是把 learned velocity field 从 `t=0` 积分到 `t=1` 得到的；
- Fisher bridge 仍然存在，但它是训练 pair `(theta0, theta1+)` 之间的解析监督路径；
- 当前主线没有 one-step endpoint student，也不把 reference bridge 当成普通推理的终点生成器。

GT 只用于初始 bootstrap、训练 replay/teacher，以及不参与选择的诊断。普通推理输入仍然只有
`(X, y, theta0)`。

## 2. 已有 smoke 足以判定 learned flow 失败

bounded GPU smoke 的 bootstrap 末尾为：

- global relative Fisher loss：`0.989`；
- readout/op/arg relative loss：`0.989 / 0.986 / 0.991`；
- predicted/target velocity norm ratio：`0.019`；
- terminal consistency：`5.16`。

这表示网络只比零速度预测器好约 1%，而预测速度范数只有目标的约 2%。它没有学会把先验分布
实质性地推向 sharp expression endpoints。

CPU smoke 给出相同结论：

- source-to-reference Fisher cost 只有约 `7.8e-6`，说明 ODE 几乎停在 `theta0`；
- KL reweighting 把样本内期望能量从 `0.549` 降到 `0.397`，但这是“在随机候选里重排权重”，
  不是 flow 产生了正确表达式；
- 把几乎不动的 reference endpoint 拉到 sharp cell 时，旧 correction ratio 达到约 `1.28e5`；
- eval raw R2 为 `-0.014`，skeleton/operator-dependency/GT equivalence 均为 `0`；
- terminal retraction FR mean/p95 为 `0.409 / 0.512`，远高于可接受门槛。

所以当前不能声称“从初始端点运行 flow 后采样到了 GT”。hard decode 与少量 categorical sample
主要反映先验 argmax/随机性；KL energy、fitted R2、medoid 或 retraction 后的合法表达式都不能替代
这一事实。

## 3. v4 的最小可证伪算法

每个任务使用固定 source seeds，使同一个 `source_index` 在所有 outer iterations 中代表同一个
`theta0`。每轮执行：

1. learned direct-velocity RK2 rollout 得到 `bar_theta1_i`；
2. hard decode 完整表达式 `z_i`；
3. 只在 decoded expression 的 active blocks 上计算 `bar_theta1_i` 到其自身 epsilon-sharp
   decoded cell 的 Fisher gap；inactive blocks 只检查恒等保持，不能用于稀释 gate；
4. 任务级 mean/p95 未通过 `0.15 / 0.35` 时立即拒绝本轮，完全不做表达式语义搜索；
5. 通过后，才在 active blocks 上构造少量单-block邻居，用 hard-prefix register signed-pair
   reachability 排序；
6. 只对最多 `K=6` 个完整表达式计算 raw output NMSE/signature objective；
7. 在 `FR RMS <= 0.35` 的局部 cell 中做 MAP 选择；
8. 保持原 source lineage，直接训练 `(theta0_i, theta1_i+)` 的 Fisher flow matching；不做跨 source OT。

主线明确不再使用：

- 全局 `16/64` 组表达式采样；
- KL posterior；
- Sinkhorn/Hungarian recoupling；
- mutation、elite selection、archive；
- block-marginal posterior projection。

## 4. 开销为何降到可接受范围

旧 v3 的主要昂贵部分近似为：

```text
N sources x M complete traces x register/output semantic execution
+ N x J Sinkhorn cost/iterations
```

其中 smoke/overfit 的 `M` 分别为 `16/64`，而重复表达式仍会造成大量前缀与语义计算。

v4 改为：

```text
N learned ODE rollouts
+ N x T_GT cheap compiled-trace diagnostics
+ [only after manifold gate passes]
   N x (B_active x A_alt) cheap structural proposals
   + N x K expensive raw semantic evaluations, K <= 6
```

在当前已失败的状态，manifold gate 会在语义搜索前停止，因此 expensive candidate semantic
evaluation 为零。即使以后通过 gate，昂贵部分也从 `N x 64` 降为至多 `N x 6`，且没有全局 OT。
GT categorical probe 默认仅 4 次，纯属诊断，可设为 0；它不产生训练候选。

## 5. “flow 是否真的生成 GT”的报告标准

必须同时报告下列直接量：

- `flow_hard_gt_symbolic_hit`：learned endpoint 的 argmax 完整表达式是否等价于 GT；
- `flow_sample_gt_hit_rate`：从 learned endpoint 做固定少量无 GT 采样时的 GT 命中率；
- `flow_gt_trace_probability_geometric_mean_max`：compiled GT trace active decisions 的几何平均概率；
- `flow_gt_trace_active_argmax_match_max`：GT active blocks 的 argmax 匹配率；
- `flow_nearest_gt_cell_fr_rms`：learned endpoint 到最近 compiled GT cell 的 Fisher 距离；
- `reference_manifold_fr_mean/p95`：learned endpoint 到自身 decoded cell 的距离。

只有 hard/sample GT hit 出现，或 GT trace mass/argmax 匹配持续上升且 GT-cell、own-cell 距离同时
下降，才能说明 flow 正在生成 GT basin。raw R2 和结构准确率用于确认输出质量；fitted R2、best-of-N
只允许作为明确标注的辅助诊断。

极小 CPU 接线验证在 active-only 修正后给出 own-cell mean/p95 `2.533/2.533`、hard/sample GT hit
`0/0`、compiled GT trace 几何平均概率 `0.151`、nearest-GT-cell FR RMS `1.988`，并在 expensive
semantic scoring 前停止（候选执行数 `0`）。这正是预期的失败判定，而不是性能结果。

## 6. 三排 coupling landscape 图

输出文件：

- `outer_iteration_flow_landscape.jsonl`：原始 fixed-source ODE snapshots；
- `outer_iteration_flow_coupling.svg`：论文矢量图；
- `outer_iteration_flow_coupling.png`：高分辨率位图；
- `outer_iteration_flow_coupling_meta.json`：outer iteration、ODE time 与 PCA 方差元数据。

图固定为三排：首轮、outer middle、最终轮。每个 ODE 时间点有相邻两幅小图：

- 参数空间：对 `2*sqrt(p)` 的 Hellinger/Fisher chart 做统一 PCA；
- 表达式空间：对 hard-decoded raw output semantic signature 做统一 PCA。

同一 source 在所有轮次使用相同颜色。实线表示 learned ODE trajectory，空心圆表示 `theta0`，
菱形表示通过 gate 后接受的 local proximal target，星号表示最近 compiled GT cell（只做诊断）。
终点到局部目标的虚线与到 GT cell 的点线被明确分开，因此图不会用 terminal retraction 掩盖
learned flow 没有到达目标的事实。
