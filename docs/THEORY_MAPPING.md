# Theory To Code Mapping

Current theory:

```text
Edge-Parameterized Semantic Flow Matching for Symbolic Regression
```

The probability object is a task-conditioned edge distribution `Theta`, not a
local action policy. A sampled point from `Theta` is a complete executable DAG.

| Theory object | Symbol | Code |
|---|---|---|
| Dataset/task | `D=(X,y)` | `semflow_sr/edge_flow/dataset.py`, synthetic smoke generator |
| Circuit template | `G={C_g}` | `semflow_sr/edge_flow/template.py::RegisterOperatorTemplate` |
| Edge group | `C_g` | `EdgeGroup` with `ARG_SELECT`, `REG_UPDATE`, `OUTPUT_SELECT` |
| Edge distribution | `Theta=(alpha, theta)` | `semflow_sr/edge_flow/edge_distribution.py::EdgeDistribution` |
| Complete circuit sample | `z=(h,z_1,...,z_N)` | `semflow_sr/edge_flow/circuit_sampler.py::CircuitSample` |
| Expression map | `e=pi(z)` | `CircuitSampler._build_expression` |
| Complete-expression reward | `R_D(e)` | `semflow_sr/edge_flow/reward.py::RewardEvaluator` |
| Empirical elite target | `pi_hat_D` | top-k elites in `projection.py` |
| Edge target projection | `Theta*_D` | `project_elites_to_edge_target` |
| Fisher sqrt path | `z_lambda=sqrt(theta_lambda)` | `flow_teacher.py::build_fisher_slerp_record` |
| Velocity target | `dot z_lambda` | `EdgeFlowRecord.group_zdot`, `mixture_zdot` |
| Learned vector field | `V_psi(D,Theta,lambda)` | `model.py::EdgeFlowModel` |
| Inference integration | `Theta0 -> Theta1` | `scripts/run_edge_flow.py::_integrate` |

## Current Smoke Workflow

```text
configs/train/edge_flow_smoke.yaml
-> semflow_sr.edge_flow.train_edge_flow
-> checkpoints/edge_flow_smoke.pt
-> scripts/run_edge_flow.py
-> results/edge_flow_smoke/
```

## Legacy Mapping

The old action-level Semantic-Fisher code is still present under
`semflow_sr/path_posterior/` for regression comparison. It is not the main
theory mapping and should not be used to describe the current algorithm.
