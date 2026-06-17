import torch
from semflow_sr.registers.state import init_register_state
from semflow_sr.registers.executor import evaluate_register_state
from semflow_sr.actions.action_space import ActionSpace, ActionSpec
from semflow_sr.actions.action_executor import ActionExecutor
from semflow_sr.semantics.energy import ActionEnergy, ActionEnergyConfig


def test_semantic_equals_symbolic_execution():
    torch.manual_seed(0)
    K = 6
    state = init_register_state(num_vars=2, K=K)
    space = ActionSpace(K)
    execu = ActionExecutor(space)
    X = torch.randn(32, 2)
    B = evaluate_register_state(state, X)

    # apply mul(reg0, reg1) -> write reg3
    spec = ActionSpec(op_id=2, read_1=0, read_2=1, write=3)
    aid = space.encode(spec)

    # symbolic path
    s2 = execu.execute_symbolic(state, aid)
    B_sym = evaluate_register_state(s2, X)

    # semantic path
    B_sem = execu.execute_semantic(B, torch.tensor([aid]))[0]

    assert torch.allclose(B_sym, B_sem, atol=1e-5)


def test_action_energy_evaluate_actions_matches_compute_and_rewards():
    torch.manual_seed(1)
    state = init_register_state(num_vars=1, K=5)
    space = ActionSpace(K=5, allowed_ops=[0, 1, 2, 7])
    X = torch.linspace(-1, 1, 32).unsqueeze(1)
    y = X.squeeze() + X.squeeze() ** 2
    B = evaluate_register_state(state, X)
    ids = space.valid_actions(state)[:16]
    energy = ActionEnergy(space, ActionEnergyConfig(lambda_op=0.01))

    ev = energy.evaluate_actions(B, y, ids)

    assert torch.allclose(ev.energies, energy.compute(B, y, ids))
    assert torch.allclose(ev.rewards, energy.rewards(B, y, ids))
    assert ev.B_after.shape == (ids.numel(), X.shape[0], state.K)


def test_action_semantic_effects_use_same_centered_projection_backend():
    torch.manual_seed(7)
    state = init_register_state(num_vars=1, K=5)
    space = ActionSpace(K=5, allowed_ops=[0, 1, 2, 7])
    X = torch.linspace(-1, 1, 32).unsqueeze(1)
    y = X.squeeze() + X.squeeze() ** 2
    B = evaluate_register_state(state, X)
    ids = space.valid_actions(state)[:12]
    cfg = ActionEnergyConfig(lambda_op=0.01)
    energy = ActionEnergy(space, cfg)

    effect = energy.action_semantic_effects(B, y, ids)

    assert effect.residual_current.shape == (X.shape[0],)
    assert effect.residual_next.shape == (ids.numel(), X.shape[0])
    assert effect.xi.shape == (ids.numel(), X.shape[0])
    assert effect.gram.shape == (ids.numel(), ids.numel())
    assert torch.allclose(
        effect.xi,
        effect.residual_current.unsqueeze(0) - effect.residual_next,
        atol=1e-10,
    )
    assert torch.allclose(effect.gram, effect.xi @ effect.xi.transpose(-1, -2), atol=1e-10)
    expected_rewards = (
        0.5 * (effect.residual_current.square().sum() - effect.residual_next.square().sum(dim=-1))
        - cfg.lambda_op * effect.op_costs
    )
    assert torch.allclose(effect.rewards, expected_rewards, atol=1e-8)
    assert torch.allclose(effect.rewards, energy.rewards(B, y, ids), atol=1e-8)
