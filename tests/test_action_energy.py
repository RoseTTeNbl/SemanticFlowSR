import torch
from semflow_sr.registers.state import init_register_state
from semflow_sr.registers.executor import evaluate_register_state
from semflow_sr.actions.action_space import ActionSpace, ActionSpec
from semflow_sr.actions.action_executor import ActionExecutor


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
