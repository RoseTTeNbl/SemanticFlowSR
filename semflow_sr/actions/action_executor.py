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
        specs = [self.space.decode(int(aid)) for aid in action_ids.tolist()]
        op_ids = torch.tensor([s.op_id for s in specs], device=B.device)
        r1 = torch.tensor([s.read_1 for s in specs], device=B.device)
        r2 = torch.tensor([s.read_2 for s in specs], device=B.device)
        write = torch.tensor([s.write for s in specs], device=B.device)
        rows = torch.arange(m, device=B.device).unsqueeze(0)
        for op_id in op_ids.unique().tolist():
            idx = (op_ids == int(op_id)).nonzero(as_tuple=False).squeeze(-1)
            op = get_op(int(op_id))
            if op.arity == 1:
                col = op.fn(B[:, r1[idx]]).transpose(0, 1)       # [n,m]
            else:
                col = op.fn(B[:, r1[idx]], B[:, r2[idx]]).transpose(0, 1)
            out[idx.unsqueeze(1), rows.expand(idx.numel(), -1), write[idx].unsqueeze(1)] = col
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
