# Math spec: Poisson semantic residual Fisher flow

For task `D=(X,y)`, the public conditional field is

\[
v_k:\mathcal S_G\times\mathcal D\times[0,1]\to T\mathcal S_G,
\qquad \dot\theta_t=v_k(\theta_t,D,t).
\]

It has no source or route condition.  Independent samples
`theta_0,i ~ mu_0` therefore generate a task-conditioned endpoint population
`theta_1,i` through one shared ODE.

## Semantic tilt and Poisson transport

Each endpoint decodes one complete trace `z_i`.  An affine coefficient is fit on
a deterministic support split and evaluated on a disjoint query split:

\[
E_i=1-R^2_{\rm query}(z_i).
\]

The infinitesimal Gibbs tilt of endpoint density `rho` is

\[
\dot\rho=-(E-\mathbb E_\rho E)\rho.
\]

Among tangent fields realizing this density derivative, the minimum Fisher
kinetic-energy solution is a gradient field `g=grad_FR phi` satisfying

\[
-\operatorname{div}(\rho\,\operatorname{grad}_{\rm FR}\phi)
=-(E-\bar E)\rho.
\]

Its empirical weak objective is

\[
\widehat{\mathcal A}(\phi)=\frac1K\sum_i
\left[\frac12\|\operatorname{grad}_{\rm FR}\phi_i\|_{\rm FR}^2
+(E_i-\bar E)(\phi_i-\bar\phi)\right].
\]

For one categorical block, an ordinary covector `c` becomes the Fisher natural
gradient

\[
g_a=p_a\left(c_a-\sum_jp_jc_j\right),\qquad \sum_ag_a=0.
\]

The corrected endpoint uses the same particle index:

\[
\theta_{1,i}^+=\operatorname{Exp}_{\theta_{1,i}}
(\epsilon g(\theta_{1,i},D)).
\]

No empirical coupling optimization is required.  To first order,

\[
\frac{d}{d\epsilon}\mathbb E_{\rho_\epsilon}[E]\big|_{0}
=-\operatorname{Var}_\rho(E)\le 0.
\]

## Bridge variation and residual field

Let `gamma(theta_0,theta_1)` be the analytic product-simplex Fisher bridge.
Moving only its endpoint defines the Jacobi variation

\[
J_t=\partial_\epsilon
\gamma_t(\theta_0,\theta_1^+(\epsilon))\big|_{0}.
\]

Differentiating the Lagrangian identity
`partial_t gamma_t = v_k(gamma_t,D,t)` gives the Eulerian field correction

\[
\Delta v=\nabla_tJ-\nabla_Jv_k.
\]

The implemented finite target is

\[
b_t=\frac{u_t^+-v_k(\theta_t^+,D,t)}{\eta},
\qquad
\mathcal L_{\rm residual}=\mathbb E\|\Delta v_\psi-b_t\|_{\rm FR}^2,
\]

and the next field is

\[
v_{k+1}=v_k+\eta\Delta v_\psi.
\]

For a sufficiently small `C1` residual perturbation, the ODE flow remains
locally unique and injective.  Multimodality is represented by different source
particles entering different basins, not by averaging route labels at the same
conditioned state.
