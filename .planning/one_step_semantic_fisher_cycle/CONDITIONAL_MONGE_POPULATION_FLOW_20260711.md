# Conditional Monge Population Fisher Flow

Date: 2026-07-11

## Decision

The main field is

\[
v_\psi(\theta,D,t),
\]

not `v(theta,D,t,r)` and not `v(theta,D,t,theta0)`. The source sample `theta0` is
only the ODE initial state. Multiple expression modes are represented by different basins of
the same task-conditioned flow.

## 1. What the averaging problem actually is

For a bridge coupling, let

\[
\Theta_t=\Gamma_t(\Theta_0,\Theta_1),\qquad
U_t=\partial_t\Gamma_t(\Theta_0,\Theta_1).
\]

The conditional FM risk satisfies

\[
\mathbb E\|v-U_t\|^2
=
\mathbb E\|v-v^*\|^2
+
\mathbb E\operatorname{Var}(U_t\mid\Theta_t,D,t),
\]

where

\[
v^*(\theta,D,t)=\mathbb E[U_t\mid\Theta_t=\theta,D,t].
\]

The conditional expectation is a valid weak Eulerian velocity for the marginal probability
path. The practical failure is the second term: arbitrary endpoint assignments or crossing
bridges give conflicting teacher velocities near the same state. A finite model then learns a
smooth compromise which need not point toward any decodable expression cell.

The bridge constructor, rather than extra loss weights, must make

\[
\operatorname{Var}(U_t\mid\Theta_t,D,t)
\]

small.

## 2. Reparameterized conditional Monge bridge

For each task condition `D`, construct an interior endpoint law `rho_D_plus` and its
Fisher--Rao Monge map

\[
T_D
=
\arg\min_{T_\#\mu_0=\rho_D^+}
\mathbb E_{\theta_0\sim\mu_0}
\frac12d_{\rm FR}^2(\theta_0,T(\theta_0)).
\]

Use Fisher displacement interpolation

\[
F_{D,t}(\theta_0)
=
\operatorname{Exp}_{\theta_0}
\left(t\operatorname{Log}_{\theta_0}T_D(\theta_0)\right).
\]

When the source is absolutely continuous and the transport stays in a geodesically convex
normal neighborhood, the Monge map and interpolation are map-induced almost everywhere. Then

\[
v_D(\theta,t)
=
\partial_tF_{D,t}(F_{D,t}^{-1}(\theta))
\]

is single valued and the conditional bridge-velocity variance is zero almost everywhere.
Different initial regions are transported to different expression modes; no route label is
needed.

A finite-time Lipschitz ODE cannot push a continuous prior exactly onto finite one-hot atoms.
For expression `z`, use a continuous epsilon-sharp cell kernel

\[
K_\epsilon(d\theta\mid z)
\]

whose samples have a unique argmax decode but retain continuous base/jitter coordinates in the
simplex interior. Exact terminal retraction is evaluation-only and its Fisher jump must be
reported.

## 3. Low-cost semantic population update

For one task, sample `K=8` or `16` source particles and run the current field once per source:

\[
\bar\theta_i=\Phi_1^{v_k}(\theta_{0,i},D),
\qquad
z_i=\operatorname{HardDecode}(\bar\theta_i).
\]

There is exactly one decoded expression per endpoint. Each particle may produce at most one
local legal offspring. Cached hard register values rank edits cheaply by residual reachability;
for a candidate register value `g` and fitted residual `r`, a useful proposal score is

\[
\Delta(g)=
\frac{\langle g,r\rangle^2}{\|g\|^2+\lambda}.
\]

This score only proposes an edit. The complete parent or offspring expression is evaluated by
the same cross-fitted coefficient objective used for final selection:

\[
E_D(z)=1-R^2_{\rm crossfit,coef}(z).
\]

Complexity is only a deterministic tie-break in the first version. Raw R2 remains a required
diagnostic but does not replace coefficient-aligned selection.

Collapse duplicate canonical expressions and form the empirical proposal law `Q_tilde`. The
semantic KL-proximal update is simply

\[
Q_\beta^+(z)
\propto
Q_{\rm tilde}(z)e^{-\beta E_D(z)}.
\]

Its expected energy decreases monotonically because

\[
\frac{d}{d\beta}\mathbb E_{Q_\beta^+}[E_D]
=
-\operatorname{Var}_{Q_\beta^+}(E_D)
\le0.
\]

Choose `beta` by a fixed effective-sample-size target, initially `ESS=K/2`, instead of a
temperature grid. This single rule both limits collapse and removes task-dependent energy-scale
tuning.

Lift complete expressions, not block marginals:

\[
\rho_D^+
=
\sum_j w_jK_\epsilon(\cdot\mid z_j).
\]

Create `K` deterministic target slots from these weights, draw only cheap continuous endpoint
jitter for each slot, and solve the squared-Fisher assignment from the `K` source particles to
the `K` fixed targets. Hungarian is the finite-sample Monge approximation; no barycentric target
and no soft block average are used.

Finally train only

\[
\mathcal L_{\rm FM}
=
\mathbb E\left[
\|v_\psi(\Gamma_t(\theta_0,T_D(\theta_0)),D,t)
-
\partial_t\Gamma_t(\theta_0,T_D(\theta_0))\|_{\rm FR}^2
\right].
\]

## 4. Canonical tilt-to-velocity interpretation

For the continuously tilted endpoint density

\[
\rho_\beta\propto e^{-\beta\mathcal E_D}\rho_0,
\]

\[
\partial_\beta\rho_\beta
=
-(\mathcal E_D-\bar{\mathcal E}_D)\rho_\beta.
\]

Representing this density change by deterministic transport requires

\[
\partial_\beta\rho_\beta
+
\operatorname{div}_{\rm FR}(\rho_\beta w_\beta)=0.
\]

The minimum-kinetic-energy solution has `w_beta=grad_FR phi_beta`, where

\[
\operatorname{div}_{\rm FR}
(\rho_\beta\nabla_{\rm FR}\phi_\beta)
=
(\mathcal E_D-\bar{\mathcal E}_D)\rho_\beta.
\]

This weighted Poisson equation is the exact mathematical answer to “how semantic tilt becomes
parameter velocity.” The implementation uses finite-particle Fisher Monge assignment instead of
solving this high-dimensional PDE.

## 5. Field architecture boundary

- Remove `theta0` from all model features and signatures. It remains only the integrator start.
- Do not introduce route/source IDs.
- ODE-time features must be continuous in `theta`: task set embedding, probabilities, and soft
  register-bank summaries are allowed.
- Hard argmax, hard-prefix execution, discrete add/sub edits, coefficient fitting, and population
  ranking are outer target-construction operations only.
- Cache the task embedding once per task. In the first implementation, replace per-action
  semantic tensors in every RK2 call with compact soft register summaries.
- A bounded continuous MLP is locally Lipschitz in `theta`; with a deterministic ODE solver it
  defines a unique task-conditioned flow. Spectral normalization is optional, not a new loss.

Particle competition and information sharing initially occur in the population target update.
If independent rollouts later collapse, a permutation-equivariant DeepSets population summary
can be added to define a field on the product manifold, but it is not required for the first
minimal experiment.

## 6. Training and inference contract

Training bootstrap may use compiled GT endpoints, but they must be converted into the same
continuous target law and Monge bridge. GT is not a field input. Later outer updates use only
`(X,y)`, decoded expressions, coefficient fitting, and register semantics.

Ordinary inference is strictly

\[
K\theta_0
\to
K\text{ learned-flow rollouts}
\to
K\text{ hard expressions}
\to
\text{cross-fit selection}.
\]

The main flow-only metrics are recorded before any local edit. Optional inference-time evolution
is a separately labelled enhancement, not part of the claim that the learned flow recovers the
expression.

Generalization to arbitrary unseen tasks cannot be guaranteed without assumptions. The precise
learnable object is stable only if the semantic target constructor `D -> rho_D_plus` varies
continuously over the task distribution. Training should therefore use a permutation-invariant
task encoder, random observation subsets, identical normalization, and the same coefficient-fit
semantic rule available at inference.

## 7. Diagnostics before another medium run

Do not add these as losses initially; record them:

1. nearest-neighbor conditional bridge-velocity variance ratio, split by low/mid/high `t`;
2. empirical cyclic-monotonicity violation rate of the source-target assignment;
3. target-expression versus rollout-expression agreement;
4. flow-only population raw/cross-fit R2 median, best, and selected values;
5. pre/post-retraction expression agreement and Fisher jump;
6. wall time separated into rollout, semantic child generation, assignment, FM training, and
   evaluation.

The next run should first be an 8-task short diagnostic. A medium run is justified only after a
GT-bootstrap overfit test shows that the task-only field can recover its continuous target cells
without large retraction and without high low-t conditional velocity variance.

## 8. Deliberately rejected first-version alternatives

- Route or `theta0` conditioning: avoids conflict by changing the mathematical object.
- Direct reward-weighted FM: reweights paths but does not create missing expression support.
- Direct blockwise replicator drift: cheap, but can destroy full-trace correlation.
- SVGD: hard expression scores require another differentiable critic.
- Schrödinger bridge/control-as-inference: elegant but introduces diffusion/score or continuation
  estimation and exceeds the current runtime budget.
- Entropic/barycentric endpoint OT: preserves soft ambiguity and can recreate blurred endpoints.

The first implementation should therefore remain:

```text
K theta0
  -> K learned rollouts
  -> K complete expressions
  -> <=1 semantic local offspring per particle
  -> complete-expression Gibbs tilt (ESS controlled)
  -> K continuous sharp target slots
  -> Fisher Monge/Hungarian assignment
  -> one Fisher FM loss
```
