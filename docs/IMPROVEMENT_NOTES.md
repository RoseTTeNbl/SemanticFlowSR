# Improvement Notes

Current mainline:

```text
Edge-Parameterized Semantic Flow Matching
```

The action-level TSSF/SFFM direction is now legacy. Its main theoretical
failure was not only target construction noise; the probability object itself
was local:

```text
state s_t -> action simplex A_s -> commit one action
```

That structure encouraged building overcomplete register dictionaries and then
using dense readout-like fitting to recover high R2. It did not directly model
a distribution over compact complete expressions.

---

## 1. New Diagnosis

The algorithm should search over complete executable expressions. The new
object is:

```text
Theta -> q_Theta(e)
```

where `Theta` is a low-dimensional product of edge-choice simplexes and
`q_Theta` is the induced distribution over full expression DAGs.

This changes the failure surface:

```text
old bottleneck: local support / action target / local teacher imitation
new bottleneck: template coverage / expression sampling / elite projection / edge-flow imitation
```

---

## 2. Current Implementation Checkpoint

Implemented smoke stack:

```text
RegisterOperatorTemplate
EdgeDistribution
CircuitSampler
RewardEvaluator
EliteProjection
Fisher-slerp teacher
EdgeFlowModel
train_edge_flow smoke CLI
run_edge_flow smoke CLI
```

Smoke command:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/edge_flow_smoke.yaml
```

Smoke evaluation command:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python scripts/run_edge_flow.py \
  --ckpt checkpoints/edge_flow_smoke.pt \
  --out results/edge_flow_smoke \
  --tag edge_flow_smoke
```

The smoke run is not a benchmark result. It only confirms that the new data
format and flow path execute.

---

## 3. Current 87-Task Signal

The first full 87-task Edge Flow run was:

```text
configs/train/edge_flow_87_basic.yaml
results/edge_flow_87_basic/
```

Headline metrics:

```text
R2 mean            0.7931
solution rate      0.0690
skeleton accuracy  0.0230
complexity mean    3.1264
valid fraction     0.9988
unique fraction    0.6968
```

The important reading is not that the mean R2 is unusable. It is that low
complexity and high validity were achieved by choosing simple correlated
complete expressions, not by recovering symbolic structure. Only two tasks have
exact skeleton match under the current 0/1 skeleton metric:

```text
constant/Constant-5
nguyen/Nguyen-8
```

This confirms the user's observation: many outputs look qualitatively close on
some examples, but the algorithm mostly learns low-dimensional semantic
surrogates rather than the target expression skeleton.

The next diagnostic run was:

```text
configs/train/edge_flow_87_h4_l3_k8.yaml
results/edge_flow_87_h4_l3_k8/
```

It changed the structure and training target in the intended direction:

```text
H=4 mixture modes
L=3 layers
K=8 registers
per-mode top-k hard projection
validation-split robust target rewards
decoder budget curve
prior and theta_star oracle diagnostics
```

Headline metrics:

```text
R2 mean            0.8185
solution rate      0.0920
skeleton accuracy  0.0230
complexity mean    4.0460
valid fraction     0.9976
unique fraction    0.6271
```

This improves numeric fit over the basic run, but it does not improve the main
structural failure:

```text
basic H1/L2/K5:  R2 0.7931, solution 0.0690, skeleton 0.0230
H4/L3/K8:        R2 0.8185, solution 0.0920, skeleton 0.0230
```

### 3.1 What The Metrics Say

`complexity_mean=3.13` means the new complete-DAG output structure did address
the previous dense-readout over-complexity symptom. The output is no longer a
large affine combination of many register columns.

`valid_expression_fraction_mean=0.9988` means numerical protection and template
execution are not the dominant problem for this configuration.

`unique_expression_fraction_mean=0.6968` is acceptable for a smoke-level
sampling decoder, but it also shows that the final distribution still collapses
onto repeated simple structures.

`solution_rate=0.0690` and `skeleton_accuracy=0.0230` identify the real
bottleneck: the search distribution rarely puts enough mass on the correct
complete expression family. High R2 cases often come from single-term
approximations:

```text
polynomial target -> x0**3 or x0**2 surrogate
trigonometric product -> sin(x0) or cos(cos(x0)) surrogate
multi-variable target -> one-variable projection
nonlinear physical law -> low-dimensional correlated proxy
```

The subset split is also diagnostic:

```text
num_vars=1: R2 0.9219, skeleton 0.0952
num_vars=2: R2 0.8196, skeleton 0.0000
num_vars=3: R2 0.6993, skeleton 0.0000
jin:      R2 0.5808, skeleton 0.0000
feynman:  R2 0.7530, skeleton 0.0000
```

So the model can find simple one-dimensional correlates, but it is not
assembling multi-variable and multi-operator structures.

### 3.2 What The New Module Diagnostics Say

The H4/L3/K8 run added four direct bottleneck probes.

Decoder budget:

```text
sample 256:   R2 0.7707, skeleton 0.0115
sample 1024:  R2 0.8273, skeleton 0.0230
sample 4096:  R2 0.8678, skeleton 0.0230
```

Increasing the decode budget recovers better semantic fits, but it does not
recover correct skeletons. This means ordinary sampling budget is a numeric
fit bottleneck, not the primary structure bottleneck.

Prior and projection oracles:

```text
prior oracle best R2 mean       0.8491
theta_star decode best R2 mean  0.8743
theta_star projection drop     -0.0252
prior skeleton accuracy         0.0230
theta_star skeleton accuracy    0.0345
model skeleton accuracy         0.0230
```

The target projection is not destroying high-R2 elites on average. In fact,
sampling from `theta_star` improves mean best R2. However, both prior and
`theta_star` still find the same wrong structural family. Therefore the most
important failure is upstream of neural flow imitation:

```text
reward + template + sampled expression family still prefer correlated proxy
expressions over correct skeletons.
```

Mixture diagnostics:

```text
per-mode elite count total: [696, 696, 696, 696]
per-mode nonzero elite tasks: [87, 87, 87, 87]
mean per-mode best reward: approximately 0.79 to 0.80
```

Hard per-mode projection avoids empty modes in this configuration. It does not
by itself create structurally distinct modes. The modes currently discover
similar proxy families.

Reward robustness and calibration:

```text
median calibration gain               1.4676
tasks with calibration gain > 0.1      82.8%
tasks with calibration gain > 1.0      55.2%
median raw uncalibrated test R2       -0.8453
median train-test R2 gap               0.0004
```

The validation split reduced obvious train/test overfit, but it did not remove
the proxy-expression problem. Most high-scoring expressions still depend
heavily on affine calibration:

```text
single expression g(x) is often poor by itself,
but a*g(x)+b becomes a high-R2 local surrogate.
```

Variable dependency:

```text
mean used variable count           1.6092
1-variable full dependency frac    1.0000
2-variable full dependency frac    0.7241
3-variable full dependency frac    0.0270
```

The 3-variable tasks are the clearest failure mode. The final expressions
almost never use all required variables, so multi-variable physics-style
structure is not being assembled.

### 3.3 Current Diagnosis After H4/L3/K8

The bottleneck ordering is now:

```text
1. reward/output selection accepts affine-calibrated proxy expressions;
2. template/decode underuse all variables, especially in 3D tasks;
3. mixture modes do not yet specialize into different structural families;
4. neural flow imitation is secondary until theta_star skeleton improves.
```

This is different from the old action-level failure. The new method fixed the
dense-readout complexity symptom, but it now searches too comfortably inside a
space of compact correlated surrogates.

## 4. Current Bottlenecks

### 4.1 Template Expressivity Is Too Shallow

The 87-basic run uses:

```text
num_layers=2
num_registers=5
H=1
```

This keeps outputs compact, but many 87-task targets require compositions,
products of subexpressions, or multiple active terms. With the current template,
a sampled expression often becomes a single primitive chain or a simple carried
register. This explains why complexity is low but skeleton accuracy is near
zero.

Next change:

```text
Keep L/K diagnostics active, but do not blindly increase capacity. The H4/L3/K8
run shows that larger capacity improves R2 without improving skeleton. Future
template changes must specifically increase multi-term and multi-variable
composition:

  output slots over composite nodes,
  explicit top-level add/mul/div composition nodes,
  forced opportunities for all active input variables,
  dimension-specific templates.
```

### 4.2 Reward Optimizes Semantic Fit, Not Structure

Reward is currently:

```text
R2(affine_calibrated single expression) - lambda_c * complexity
```

This rewards any expression whose curve is correlated with the target on the
sampled domain. It does not distinguish:

```text
correct skeleton with poor constants
wrong skeleton with high local correlation
one-variable projection of a multi-variable target
```

The new skeleton metric shows this directly: high R2 does not imply structural
recovery.

Next change:

```text
Treat affine calibration as a diagnostic and a final refit tool, not the main
training signal. Candidate changes:

  score with a blend of raw R2 and calibrated R2,
  penalize excessive calibration_gain,
  require validation/probe robustness,
  add variable coverage and operator-family diagnostics before any reward term.
```

The auxiliary terms should be diagnostics first. Do not hard-code skeleton
reward from ground truth into training except for supervised ablation.

### 4.3 The Target Projection Is Too Crude

Top-k weighted edge counts project elite samples independently per edge group.
This breaks dependencies between edges. A correct expression is a coordinated
set of choices; marginal edge counts can average incompatible circuits into a
distribution that samples plausible but wrong fragments.

Next change:

```text
H=4 hard per-mode projection is now implemented and stable enough to run. It
does not yet produce structural mode diversity. Next projection work should
measure and preserve dependencies:

  edge co-occurrence / mutual information among elites,
  elite skeleton diversity,
  mode-specific root/operator histograms,
  replay elites by full expression, not local fragments.
```

### 4.4 Sampling Decode Is Still Weak

The current decoder samples 256 complete expressions from the final
distribution. That is too small for the combinatorial template, especially when
the learned distribution is diffuse or has wrong marginal dependencies.

Next change:

```text
Sample-budget curves are now recorded. They show numeric R2 improves strongly
with budget, while skeleton does not. Beam decoding is still useful, but it is
not expected to solve skeleton accuracy unless the target distribution changes.

Next decoder work:

  add beam decode for high-probability circuits,
  report reward-only vs reward + eta log q selection,
  keep budget curves as a standard output file.
```

### 4.5 Single d=3 Checkpoint Is A Compromise

The 87-basic run pads 1D/2D tasks into a single d=3 template. This is convenient
for one configuration group, but it confounds:

```text
true variable structure
dummy variable edges
template candidate counts
data encoder statistics
```

Next change:

```text
Train d1/d2/d3 Edge Flow checkpoints or make the template dynamically sized.
Compare per-dimension checkpoints against padded d=3.
```

## 5. Key New Bottlenecks To Measure

### 5.1 Template Coverage

The template defines the reachable expression family. Record:

```text
num_layers
num_registers
primitive set
num edge groups
candidate count distribution
```

If the template cannot express a task compactly, flow quality cannot fix it.

### 5.2 Sampling Coverage

For every task:

```text
num_sampled_expressions
valid_expression_fraction
unique_expression_fraction
duplicate_expression_fraction
best_reward
median_reward
average_complexity
used_variable_count
output_depth
active_operator_histogram
```

If uniform/prior sampling never sees useful expressions, elite projection will
be biased.

### 5.3 Elite Projection Quality

Record:

```text
elite_k
target_ESS
target_edge_entropy_mean
mode_entropy
per_mode_elite_count
per_mode_best_reward
edge_target_entropy_by_group_type
elite_skeleton_diversity
```

These replace old `full_best_in_support` and `support_best_reward_gap`
diagnostics because local action support is gone.

### 5.4 Flow Imitation

Record:

```text
loss_mixture
loss_groups
velocity_norm
simplex_mass_error
endpoint distance
```

If elite targets are good but model integration cannot reproduce them, the
bottleneck is model capacity or edge-state features.

---

### 5.5 Structural Metrics

The evaluation now records:

```text
gt_skeleton
pred_skeleton
skeleton_match
skeleton_accuracy
```

The skeleton metric ignores numeric constants and affine calibration, strips
protected `Abs`, and preserves variables plus discrete exponent/operator
structure. It is intentionally a strict 0/1 structural check.

Future structural diagnostics should add softer variants:

```text
operator multiset F1
variable-set accuracy
tree edit distance
subtree recall
term-count error
```

## 6. Immediate Next Steps

1. Add a raw-vs-calibrated reward ablation:
   `R = raw_R2 + alpha * calibrated_R2 - beta * calibration_gain - lambda_c*C`.
2. Add dimension-specific Edge Flow templates/checkpoints for d1/d2/d3.
3. Add beam decode and compare against sample 4096 on the existing H4/L3/K8
   checkpoint.
4. Add variable-dependency diagnostics to projection elites, not only final
   selected expressions.
5. Add edge co-occurrence / mutual information diagnostics for top elites.
6. Add replay elites as complete DAGs with validation reward and skeleton
   diversity metadata.
7. Only after theta_star skeleton improves, spend effort on larger neural model
   capacity or semantic geometry.

---

## 7. Retired Result Interpretation

The old `results/sffm_87_20260620/` and
`results/external_baselines_87_20260620/` directories were removed from the
active result layout. Those numbers belonged to the action-level mainline and
should not be mixed with Edge Flow results.

External baselines remain relevant for future comparisons, but they should be
rerun or copied into a new result matrix once Edge Flow has a real 87-task
configuration.
