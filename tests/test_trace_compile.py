import torch
from semflow_sr.sr.parser import parse_formula
from semflow_sr.sr.ast import eval_expr
from semflow_sr.registers.compiler import compile_expr
from semflow_sr.registers.executor import evaluate_register_state
from semflow_sr.sr.ops import default_op_subset, NAME_TO_ID


def test_trace_compile_reconstructs_expression():
    expr = parse_formula("x0*x0*x0", ["x0"])
    allowed = [NAME_TO_ID[o] for o in default_op_subset()]
    trace = compile_expr(expr, num_vars=1, K=8, allowed_ops=allowed)
    assert trace is not None and len(trace) >= 1
    X = torch.linspace(-1, 1, 64).unsqueeze(1)
    target = eval_expr(expr, X)
    B = evaluate_register_state(trace.final_state, X)
    assert torch.allclose(B[:, trace.target_register], target, atol=1e-5)


def test_trace_steps_are_valid_actions():
    expr = parse_formula("sin(x0)*x0", ["x0"])
    allowed = [NAME_TO_ID[o] for o in default_op_subset()]
    trace = compile_expr(expr, num_vars=1, K=8, allowed_ops=allowed)
    assert trace is not None
    # each step's action must be in the valid support of the state before it
    from semflow_sr.actions.action_space import ActionSpace
    space = ActionSpace(8, allowed)
    for step in trace.steps:
        assert space.valid_mask(step.state)[step.action_id]


def test_default_ops_compile_add_then_unary_compositions():
    allowed = [NAME_TO_ID[o] for o in default_op_subset()]
    cases = [
        "x0 + x0**2",
        "sin(x0 + x0**2)",
        "sin(x0**2)",
        "cos(x0)",
        "sin(x0**2) + cos(x0)",
    ]
    X = torch.linspace(-1, 1, 64).unsqueeze(1)
    for formula in cases:
        expr = parse_formula(formula, ["x0"])
        trace = compile_expr(expr, num_vars=1, K=12, allowed_ops=allowed)
        assert trace is not None, formula
        target = eval_expr(expr, X)
        B = evaluate_register_state(trace.final_state, X)
        assert torch.allclose(B[:, trace.target_register], target, atol=1e-5), formula
