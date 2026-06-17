"""Per-action feature vectors for the action encoder.

Features for action a=(op,r1,r2,w): op one-hot proxy via id, arity, normalized register
ids, and register metadata (active/depth/complexity/age) of read/write registers.
"""
from __future__ import annotations
import torch

from ..sr.ops import get_op, N_OPS
from ..registers.state import RegisterState
from .action_space import ActionSpace

# feature layout: [op_id_norm, arity_norm, r1_norm, r2_norm, w_norm,
#                  active_r1, depth_r1, cplx_r1, age_r1,
#                  active_r2, depth_r2, cplx_r2, age_r2,
#                  active_w,  depth_w,  cplx_w,  age_w]
ACTION_FEATURE_DIM = 5 + 3 * 4
SEMANTIC_ACTION_FEATURE_DIM = 8


def action_features(space: ActionSpace, state: RegisterState, action_ids: torch.Tensor) -> torch.Tensor:
    K = space.K
    feats = []
    depth = state.depth.float(); cplx = state.complexity.float(); age = state.age.float(); act = state.active.float()
    dn = depth.clamp(min=1).max(); cn = cplx.clamp(min=1).max(); an = age.clamp(min=1).max()
    for aid in action_ids.tolist():
        spec = space.decode(int(aid))
        arity = get_op(spec.op_id).arity
        def reg(i):
            return [act[i].item(), (depth[i] / dn).item(), (cplx[i] / cn).item(), (age[i] / an).item()]
        row = [spec.op_id / max(N_OPS - 1, 1), (arity - 1) / 1.0,
               spec.read_1 / max(K - 1, 1), spec.read_2 / max(K - 1, 1), spec.write / max(K - 1, 1)]
        row += reg(spec.read_1) + reg(spec.read_2) + reg(spec.write)
        feats.append(row)
    return torch.tensor(feats, dtype=torch.float32, device=action_ids.device)
