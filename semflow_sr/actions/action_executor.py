"""Action execution: semantic (numeric, vectorized) and symbolic (for rollout)."""
from __future__ import annotations
import torch

from ..sr.ops import get_op
from ..sr.ast import Expr
from ..registers.state import RegisterState
from .action_space import ActionSpace, ActionSpec


class ActionExecutor:
    def __init__(self, action_space: ActionSpace):
        self.space = action_space

    def execute_semantic(self, B: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        """B:[m,K], action_ids:[A] -> B_after:[A,m,K].

        Column `write` is replaced by op(B[:,r1], B[:,r2]); other columns copied."""
        m, K = B.shape
        A = action_ids.shape[0]
        out = B.unsqueeze(0).expand(A, m, K).clone()
        # group by op_id to vectorize the protected op application
        ids = action_ids.tolist()
        for idx, aid in enumerate(ids):
            spec = self.space.decode(int(aid))
            op = get_op(spec.op_id)
            if op.arity == 1:
                col = op.fn(B[:, spec.read_1])
            else:
                col = op.fn(B[:, spec.read_1], B[:, spec.read_2])
            out[idx, :, spec.write] = col
        return out

    def execute_symbolic(self, state: RegisterState, action_id: int) -> RegisterState:
        spec = self.space.decode(int(action_id))
        op = get_op(spec.op_id)
        new = state.clone()
        if op.arity == 1:
            child = (new.exprs[spec.read_1],)
        else:
            child = (new.exprs[spec.read_1], new.exprs[spec.read_2])
        new_expr = Expr.op(spec.op_id, child)
        new.exprs[spec.write] = new_expr
        new.active[spec.write] = 1.0
        new.depth[spec.write] = new_expr.depth
        new.complexity[spec.write] = new_expr.complexity
        new.age = new.age + 1
        new.age[spec.write] = 0
        return new
