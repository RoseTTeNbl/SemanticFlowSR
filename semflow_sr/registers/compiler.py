"""Compile a full Expr into a fixed-register trace (postorder, writing into free slots).

Leaves (vars/consts seeded at init) are reused; each internal op node becomes one action
writing into the next free register. Returns a RegisterTrace whose ground-truth actions
reconstruct the expression. Used both for synthetic traces and NeSymReS conversion.
"""
from __future__ import annotations

from ..sr.ast import Expr
from ..sr.ops import get_op
from .state import RegisterState, init_register_state
from .trace import RegisterTrace, TraceStep


def compile_expr(expr: Expr, num_vars: int, K: int, allowed_ops: list[int] | None = None) -> RegisterTrace | None:
    """Returns a RegisterTrace or None if it does not fit in K registers."""
    # imported lazily to avoid the registers <-> actions package import cycle
    from ..actions.action_space import ActionSpace, ActionSpec
    from ..actions.action_executor import ActionExecutor
    space = ActionSpace(K, allowed_ops)
    execu = ActionExecutor(space)
    state = init_register_state(num_vars, K)

    # map leaf Expr -> register index (vars at 0..num_vars-1, const 1.0 at num_vars)
    def leaf_reg(e: Expr) -> int | None:
        if e.kind == "var":
            return e.var_index
        if e.kind == "const" and abs(float(e.value) - 1.0) < 1e-9:
            return num_vars
        return None

    trace = RegisterTrace(num_vars=num_vars)
    next_free = [num_vars + 1]  # first writable scratch register

    def emit(e: Expr) -> int | None:
        nonlocal state
        lr = leaf_reg(e)
        if lr is not None:
            return lr
        if e.kind == "const":          # arbitrary constants unsupported in milestone-1 compiler
            return None
        child_regs = []
        for c in e.children:
            r = emit(c)
            if r is None:
                return None
            child_regs.append(r)
        if next_free[0] >= K:
            return None
        w = next_free[0]; next_free[0] += 1
        op = get_op(e.op_id)
        r1 = child_regs[0]
        r2 = child_regs[1] if op.arity == 2 else 0
        spec = ActionSpec(e.op_id, r1, r2, w)
        aid = space.encode(spec)
        trace.steps.append(TraceStep(state=state.clone(), action_id=aid, write=w))
        state = execu.execute_symbolic(state, aid)
        return w

    final = emit(expr)
    if final is None:
        return None
    trace.final_state = state
    trace.target_register = final
    return trace
