# 理论到代码映射

## 核心对象

| 理论对象 | 符号 | 代码 |
|---|---|---|
| 局部条件 | `c=(B,y,S,p_start)` | `trace_dataset.py`, `targets/base.py::LocalCondition` |
| centered residual | `e_c = Cy - Pi_{CB,rho} Cy` | `ProjectionBackend.residual_vector` |
| centered energy | `E(B)=1/2||e_c||^2` | `ActionEnergy.compute`, `ProjectionBackend.residual_energy` |
| action semantic effect | `xi_c(a)=e_c-e_c^a` | `ActionEnergy.action_semantic_effects` |
| semantic Gram | `K_c(a,b)=<xi_c(a),xi_c(b)>` | `SemanticEffectOutput.gram` |
| scalar score | `R(a)` | one-step / rollout / search / GP providers |
| normalized advantage | `A(a)` | provider context `advantages` |
| semantic-Fisher metric | `g^SF = g^FR + gamma g^sem` | implemented through `semantic_fisher_lograte` |
| exact log-rate target | `M w = beta (A + nu 1)` | `flow/semantic_fisher.py::semantic_fisher_lograte` |
| simplex tangent | `p_dot = p ⊙ w` | `semantic_fisher_simplex_velocity` |
| sphere tangent | `z_dot = 1/2 z ⊙ w` | `semantic_fisher_sphere_velocity` |
| sphere update | `z_next = Retr(z + dt z_dot)` | `semantic_fisher_sphere_step` |
| model output | `w_theta(c,a)` | `SemanticTransformer(output_mode="semantic_fisher_lograte")` |
| 主损失 | `||z_dot_theta-z_dot_target||^2` | `SemanticFisherVelocityLoss` |

## 当前主训练链

```text
build_dataset
-> VelocityTraceDataset(path_name="semantic_fisher_pullback")
-> collate_velocity
-> SemanticTransformer(output_mode="semantic_fisher_lograte")
-> SemanticFisherVelocityLoss
```

## 当前主推理链

```text
rollout_velocity(integration_method="semantic_fisher_sphere")
-> model.lograte_logits
-> semantic_fisher_sphere_step
```

## Provider 与扩展

| 层 | 作用 | 代码 |
|---|---|---|
| one-step | 当前 residual energy decrease | `semantics/energy.py`, `targets/one_step_advantage.py` |
| rollout | 未来完成质量估计 | `rollout/`, `targets/rollout_advantage.py` |
| search | beam / search score | `targets/search_advantage.py` |
| GP | 隐式分布接口 | `gp_distill/`, `targets/gp_implicit_target.py` |

这些模块只负责提供 `R(a)` 或兼容的 target 分布。主几何始终由 `xi / gram / gamma` 决定。

## 兼容 / Ablation

| 对象 | 说明 |
|---|---|
| `natural_path.py::natural_path_from_potential` | plain Fisher exponential path ablation |
| `SpherePathLoss` | plain Fisher sphere-path training ablation |
| `closed_form_policy_update` | old endpoint update ablation |
| `gamma=0` in `semantic_fisher_lograte` | no-pullback semantic-Fisher ablation |
