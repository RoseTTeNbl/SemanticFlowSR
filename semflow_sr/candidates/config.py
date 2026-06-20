"""Configuration helpers for complete-trajectory candidate pools."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..actions.action_space import ActionSpace
from ..semantics.energy import ActionEnergyConfig
from .base import SemanticCandidate
from .sampler import (
    ActionCandidateSampler,
    BlockCandidateSampler,
    CandidateSamplerGroup,
    FullCandidateSampler,
    TrajectoryCandidateSampler,
)


SUPPORTED_BLOCK_SIZES = ("H1", "H3", "full")


@dataclass(frozen=True)
class CandidateCacheConfig:
    enabled: bool = False
    path: str = "data/candidate_targets/candidate_targets.jsonl"
    read: bool = True
    write: bool = True


@dataclass(frozen=True)
class CandidateGPriorConfig:
    enabled: bool = False
    weight: float = 1.0
    complexity_weight: float = 0.0
    source: str | None = None


@dataclass
class CandidateTrajectoryConfig:
    """Main candidate-pool config.

    The supported main experiment groups are deliberately narrow:
    Action-H1, Block-H3, and full trajectory/expression candidates.
    """

    block_sizes: tuple[str | int, ...] = SUPPORTED_BLOCK_SIZES
    budgets: dict[str, int | None] = field(default_factory=lambda: {"H1": 64, "H3": 64, "full": 64})
    block_first_topk: int = 8
    block_branch_topk: int = 4
    full_candidates: list[SemanticCandidate] = field(default_factory=list)
    cache: CandidateCacheConfig = field(default_factory=CandidateCacheConfig)
    gp_prior: CandidateGPriorConfig = field(default_factory=CandidateGPriorConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CandidateTrajectoryConfig":
        raw = dict(raw or {})
        cache_raw = raw.pop("cache", None)
        gp_raw = raw.pop("gp_prior", None)
        if "block_sizes" in raw:
            raw["block_sizes"] = tuple(raw["block_sizes"])
        if cache_raw is not None:
            raw["cache"] = CandidateCacheConfig(**cache_raw)
        if gp_raw is not None:
            raw["gp_prior"] = CandidateGPriorConfig(**gp_raw)
        return cls(**raw)


def build_candidate_sampler(
    space: ActionSpace,
    cfg: CandidateTrajectoryConfig | dict[str, Any] | None = None,
    *,
    energy_cfg: ActionEnergyConfig | None = None,
) -> TrajectoryCandidateSampler:
    cfg = cfg if isinstance(cfg, CandidateTrajectoryConfig) else CandidateTrajectoryConfig.from_dict(cfg)
    groups: list[CandidateSamplerGroup] = []
    for item in cfg.block_sizes:
        group = normalize_candidate_group(item)
        budget = cfg.budgets.get(group)
        if group == "H1":
            sampler = ActionCandidateSampler(space)
        elif group == "H3":
            sampler = BlockCandidateSampler(
                space,
                horizon=3,
                first_topk=cfg.block_first_topk,
                branch_topk=cfg.block_branch_topk,
                energy_cfg=energy_cfg,
            )
        elif group == "full":
            sampler = FullCandidateSampler(precomputed=cfg.full_candidates)
        else:
            raise ValueError(f"unsupported candidate group: {group}")
        groups.append(CandidateSamplerGroup(group, sampler, budget))
    return TrajectoryCandidateSampler(groups)


def normalize_candidate_group(value: str | int) -> str:
    if isinstance(value, int):
        if value == 1:
            return "H1"
        if value == 3:
            return "H3"
        raise ValueError("candidate block size must be 1, 3, or 'full'")
    key = str(value)
    normalized = key.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "1": "H1",
        "h1": "H1",
        "action": "H1",
        "actionh1": "H1",
        "3": "H3",
        "h3": "H3",
        "block": "H3",
        "blockh3": "H3",
        "full": "full",
        "expression": "full",
        "expr": "full",
    }
    if normalized not in aliases:
        raise ValueError("candidate block size must be one of H1, H3, or full")
    return aliases[normalized]
