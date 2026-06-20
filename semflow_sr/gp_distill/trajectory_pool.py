"""Adapters from GP final-population records to trajectory samples."""
from __future__ import annotations

from pathlib import Path
import json

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..registers.state import RegisterState
from ..trajectories.sampler import Trajectory


def load_gp_trajectory_population(
    path: str | Path,
    action_space: ActionSpace,
    initial_state: RegisterState,
    *,
    max_len: int | None = None,
) -> list[Trajectory]:
    """Load GP final population records that already contain register action ids.

    Supported formats:

    - JSONL: one object per line.
    - JSON: a list, or an object with `population` / `trajectories` / `records`.

    Each record must contain `actions`. Optional `gp_logprob`, `fitness`,
    `expression`, and arbitrary metadata are preserved.
    """
    records = _read_records(path)
    out: list[Trajectory] = []
    executor = ActionExecutor(action_space)
    for idx, raw in enumerate(records):
        actions = [int(a) for a in raw.get("actions", [])]
        if max_len is not None:
            actions = actions[: max(int(max_len), 0)]
        if not actions:
            continue
        state = initial_state.clone()
        masks = []
        prefix_states = [state.clone()]
        for action in actions:
            masks.append(action_space.valid_mask(state))
            state = executor.execute_symbolic(state, action)
            prefix_states.append(state.clone())
        metadata = dict(raw.get("metadata", {}))
        metadata.update({
            "source": "gp",
            "population_index": idx,
            "initial_state": initial_state,
            "prefix_states": prefix_states,
            "final_state": state,
        })
        if "gp_logprob" in raw:
            metadata["gp_logprob"] = float(raw["gp_logprob"])
        if "fitness" in raw:
            metadata["fitness"] = float(raw["fitness"])
        out.append(Trajectory(
            actions=actions,
            masks=masks,
            logprob_base=float(raw.get("gp_logprob", raw.get("logprob_base", 0.0))),
            expr=raw.get("expression"),
            complexity=float(raw.get("complexity", len(actions))),
            metadata=metadata,
        ))
    return out


def _read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    text = path.read_text().strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    raw = json.loads(text)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("population", "trajectories", "records", "events"):
            if isinstance(raw.get(key), list):
                return raw[key]
    raise ValueError(f"unsupported GP trajectory population format: {path}")
