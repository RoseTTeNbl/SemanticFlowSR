"""One-step action space over fixed registers.

An action is a = (op_id, read_1, read_2, write). Unary ops ignore read_2 (canonicalized
to 0). Encoding is a bijective mixed-radix index over (op_id, read_1, read_2, write).
No STOP action. Stopping is by semantic energy threshold / max steps elsewhere.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch

from ..sr.ops import OPERATORS, N_OPS, get_op

if TYPE_CHECKING:
    from ..registers.state import RegisterState


@dataclass(frozen=True)
class ActionSpec:
    op_id: int
    read_1: int
    read_2: int
    write: int


class ActionSpace:
    """Bijective action <-> id over (op, r1, r2, write) with radices (N_OPS, K, K, K)."""

    def __init__(self, K: int, allowed_ops: list[int] | None = None):
        self.K = K
        # restrict to a subset of operators if requested (still encode over full N_OPS
        # radix so ids stay stable; validity handled by mask)
        self.allowed_ops = set(range(N_OPS) if allowed_ops is None else allowed_ops)

    @property
    def size(self) -> int:
        return N_OPS * self.K * self.K * self.K

    def encode(self, spec: ActionSpec) -> int:
        K = self.K
        r2 = 0 if get_op(spec.op_id).arity == 1 else spec.read_2
        return ((spec.op_id * K + spec.read_1) * K + r2) * K + spec.write

    def decode(self, action_id: int) -> ActionSpec:
        K = self.K
        write = action_id % K; action_id //= K
        r2 = action_id % K; action_id //= K
        r1 = action_id % K; action_id //= K
        op_id = action_id
        if get_op(op_id).arity == 1:
            r2 = 0
        return ActionSpec(op_id, r1, r2, write)

    def valid_mask(self, state: RegisterState) -> torch.Tensor:
        """Boolean mask [size]. 基扩张(append)语义: write 只能写 INACTIVE 槽, 使已建好的
        基单调累积、不被覆盖。读寄存器须 active; 一元算子 read_2 规范化(==0)。
        一元算子不可作用于纯常数列(否则 exp(c)/sqrt(c) 等只生成退化常数, 浪费 append 槽)。
        所有槽都激活时无合法动作(rollout 据此停机)。"""
        K = self.K
        active = state.active.bool()
        is_const = [e.kind == "const" for e in state.exprs]   # 常数列, 禁止一元算子作用
        free = [w for w in range(K) if not active[w]]         # 仅空闲槽可写
        mask = torch.zeros(self.size, dtype=torch.bool)
        if not free:
            return mask
        for op_id in self.allowed_ops:
            arity = get_op(op_id).arity
            for r1 in range(K):
                if not active[r1]:
                    continue
                if arity == 1:
                    if is_const[r1]:                          # 一元算子作用于常数列 -> 退化, 跳过
                        continue
                    for w in free:
                        mask[self.encode(ActionSpec(op_id, r1, 0, w))] = True
                else:
                    for r2 in range(K):
                        if not active[r2]:
                            continue
                        for w in free:
                            mask[self.encode(ActionSpec(op_id, r1, r2, w))] = True
        return mask

    def valid_actions(self, state: RegisterState) -> torch.Tensor:
        """Return the support: 1-D long tensor of valid action ids."""
        return self.valid_mask(state).nonzero(as_tuple=False).squeeze(-1)
