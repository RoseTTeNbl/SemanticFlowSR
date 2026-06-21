# Algorithm: Edge-Parameterized Semantic Flow Matching

Current mainline:

```text
Edge-Parameterized Semantic Flow Matching for Symbolic Regression
```

The probability object is no longer a local action distribution
`p(a_t | s_t)`. The mainline now models a distribution over complete executable
expressions through a low-dimensional product of edge-choice simplexes.

```text
data D=(X,y)
-> register-operator circuit template
-> edge distribution Theta over complete expression DAGs
-> sample complete expressions from q_Theta
-> evaluate rewards
-> project elite expressions to target edge distribution Theta*
-> train flow Theta0 -> Theta*
-> infer by integrating Theta and sampling/decoding complete expressions
```

The old action-support Semantic-Fisher path remains in the repository only as a
legacy diagnostic. It is not the current algorithmic mainline.

---

## 1. Template

Implemented in:

```text
semflow_sr/edge_flow/template.py
```

`RegisterOperatorTemplate` defines a finite sparse DAG family:

```text
num_layers L
num_registers K
primitive set
mixture_modes H
edge-choice groups G
```

The first implementation uses fixed primitive nodes per layer. It samples:

```text
ARG_SELECT     primitive argument slots choose previous registers
REG_UPDATE     next-layer registers choose carry registers or primitive images
OUTPUT_SELECT  final output chooses a terminal register
```

This replaces local action support:

```text
old: A_s = legal one-step actions at partial state s
new: C_g = candidates for edge-choice group g in a full circuit template
```

---

## 2. Edge Distribution

Implemented in:

```text
semflow_sr/edge_flow/edge_distribution.py
```

For each mixture mode `h` and edge group `g`:

```text
theta_g^(h) in simplex(C_g)
alpha in simplex(H)
```

The distribution over a sampled circuit `z` is:

```text
q_Theta(z) = alpha_h * product_g theta_g^(h)(z_g)
```

The circuit executes to a complete expression:

```text
e = pi(z)
```

So `Theta` induces a distribution over complete expressions without enumerating
the exponentially large expression set.

The first smoke configuration uses:

```text
H = 1
uniform edge prior
```

The current 87-task diagnostic configuration uses:

```text
H = 4
mode-stratified sampling
hard per-mode elite projection
```

---

## 3. Complete Expression Sampling

Implemented in:

```text
semflow_sr/edge_flow/circuit_sampler.py
```

Sampling returns complete DAG samples:

```text
mode h
edge choices z_g for every group
expression tree
log probability
complexity
```

Mixture sampling is stratified: each mode receives samples instead of relying on
early mixture probabilities to discover every mode.

Execution uses existing expression AST and protected numeric operators:

```text
semflow_sr/sr/ast.py
semflow_sr/sr/ops.py
```

---

## 4. Reward

Implemented in:

```text
semflow_sr/edge_flow/reward.py
```

Reward is defined on complete expressions:

```text
R_D(e) = R2(affine_calibrated e; D) - lambda_c * complexity(e)
```

Affine calibration is single-expression calibration:

```text
y ~= a * e(X) + b
```

It is not dense dictionary readout. Invalid numerical expressions get a large
negative reward and remain diagnostic samples.

Training records can optionally use validation-robust projection rewards:

```text
R_target(e) = min(R2_train(e), R2_val(e)) - lambda_c * complexity(e)
```

This is enabled by `validation_fraction > 0` in the training config. Evaluation
still records raw R2, affine-calibrated R2, calibration gain, train/test gap,
and final test R2.

---

## 5. Elite Projection

Implemented in:

```text
semflow_sr/edge_flow/projection.py
```

Given sampled expressions and rewards, the first target estimator uses top-k
elite weights:

```text
omega_i = 1/k for top-k valid samples
```

For product edge distributions, each group target is a weighted edge count:

```text
theta*_g,k = sum_i omega_i * 1[choice_i[g] = k]
```

For mixture mode `H > 1`, the first implementation uses hard mode assignment:

```text
sample belongs to the mode it was sampled from
```

Each mode receives its own edge counts. Empty modes fall back to the prior.
Target smoothing keeps every simplex in the interior:

```text
theta* <- (1 - eps) * theta*_counts + eps * theta0
```

The projection diagnostics record target entropy, target ESS, per-mode elite
counts, and per-mode best rewards.

---

## 6. Flow Teacher

Implemented in:

```text
semflow_sr/edge_flow/flow_teacher.py
```

The first teacher path uses groupwise Fisher square-root interpolation:

```text
z = sqrt(theta)
z_lambda = slerp(sqrt(theta0), sqrt(theta*), lambda)
dot z_lambda = analytic slerp derivative
```

This gives a stable flow-matching target on each categorical simplex. Mixture
weights are treated as another categorical group.

The first implementation intentionally does not use the old local
semantic-Fisher linear solve. That geometry belonged to action simplexes. The
new canonical geometry is the expression-Fisher pullback; the smoke
implementation uses product Fisher square-root coordinates.

---

## 7. Model

Implemented in:

```text
semflow_sr/edge_flow/model.py
```

`EdgeFlowModel` is a lightweight data-conditioned velocity model. Inputs:

```text
task statistics from (X,y)
current edge probabilities theta_lambda
group identity and type
candidate index
mixture mode
group entropy
```

Output:

```text
predicted dot z for each mixture/group/candidate simplex
```

The model centers its implied log-rate per simplex so the predicted
square-root velocity is tangent to the simplex sphere.

---

## 8. Training Workflow

Smoke entrypoint:

```bash
conda run -n semflow python -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/edge_flow_smoke.yaml
```

Training flow:

```text
generate synthetic tasks
build uniform Theta0
sample complete expressions
evaluate complete-expression rewards
project top-k elites to Theta*
sample lambda and build fisher-slerp teacher
train EdgeFlowModel on dot z targets
save checkpoint and train curve
```

The smoke config is deliberately small and is only a wiring check.

87-task entrypoint:

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  taskset -c 0-3 conda run -n semflow \
  python -m semflow_sr.edge_flow.train_edge_flow \
  --config configs/train/edge_flow_87_h4_l3_k8.yaml
```

---

## 9. Inference Workflow

Smoke entrypoint:

```bash
conda run -n semflow python scripts/run_edge_flow.py \
  --ckpt checkpoints/edge_flow_smoke.pt \
  --out results/edge_flow_smoke \
  --tag edge_flow_smoke
```

Inference flow:

```text
Theta = uniform prior
for flow step:
    model predicts dot z on current Theta
    update every simplex in square-root coordinates
sample complete expressions from final Theta
evaluate rewards with affine calibration
return best expression
```

No local action is committed during inference.

The benchmark runner can also write module-wise diagnostics:

```text
decoder_budget_curve
prior_best_r2 / prior_best_expression / prior_best_skeleton_match
theta_star_best_r2 / theta_star_projection_drop
template and expression structure diagnostics
calibration_gain and raw_test_r2_without_affine
```

---

## 10. Migration Map

| Old action-level concept | New edge-flow concept |
|---|---|
| partial state `s_t` | full circuit template plus edge distribution `Theta` |
| action support `A_s` | edge-choice groups `C_g` |
| TargetSampler over actions | complete-expression sampling plus elite projection |
| `q_hat(a | s)` | target edge distribution `Theta*` |
| local semantic effect `xi_s(a)` | complete expression semantics `s_D(e)` |
| semantic-Fisher local ODE | product Fisher edge-flow in sqrt coordinates |
| commit one action | sample/decode complete expression after flow |
| STOP action | output selector and finite-depth template |
| dense register readout | single sampled sparse DAG with affine calibration |

---

## 11. Current Implementation Status

Implemented:

```text
H>=1 edge distribution storage
stratified mixture sampling
complete expression DAG execution
affine-calibrated reward
validation-robust target rewards
top-k elite projection
hard mode assignment for mixtures
fisher-slerp teacher records
lightweight edge-flow model
smoke and 87-task train/eval CLIs
benchmark result writer with expressions, grouped stats, and diagnostics
```

Not yet implemented:

```text
soft EM responsibilities
semantic pullback extension gamma > 0
beam decoding
replay buffer
dimension-specific d1/d2/d3 edge-flow checkpoints
edge co-occurrence / mutual information diagnostics
```
