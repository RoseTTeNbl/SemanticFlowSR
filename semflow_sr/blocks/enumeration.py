"""Executable H-step block enumeration."""
from __future__ import annotations

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..registers.state import RegisterState


def enumerate_executable_blocks(
    action_space: ActionSpace,
    state: RegisterState,
    *,
    block_size: int,
    budget: int | None = None,
) -> list[tuple[int, ...]]:
    """Depth-first executable block enumeration with an optional deterministic cap."""
    h = max(int(block_size), 1)
    limit = None if budget is None else max(int(budget), 0)
    if limit == 0:
        return []
    executor = ActionExecutor(action_space)
    out: list[tuple[int, ...]] = []

    def dfs(current: RegisterState, prefix: tuple[int, ...]) -> None:
        if limit is not None and len(out) >= limit:
            return
        if len(prefix) == h:
            out.append(prefix)
            return
        valid = action_space.valid_actions(current).detach().cpu().tolist()
        for action_id in valid:
            next_state = executor.execute_symbolic(current, int(action_id))
            dfs(next_state, prefix + (int(action_id),))
            if limit is not None and len(out) >= limit:
                return

    dfs(state.clone(), tuple())
    return out


def block_table_mask(blocks: list[tuple[int, ...]], *, block_size: int, action_vocab_size: int):
    import torch

    mask = torch.zeros(int(block_size), int(action_vocab_size), dtype=torch.bool)
    for block in blocks:
        for h, action_id in enumerate(block[: int(block_size)]):
            if 0 <= int(action_id) < int(action_vocab_size):
                mask[h, int(action_id)] = True
    return mask

