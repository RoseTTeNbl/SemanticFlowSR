"""JSONL cache for trajectory target records."""
from __future__ import annotations

from ..utils.jsonl_cache import JsonlRecordCache


class TrajectoryTargetCache(JsonlRecordCache):
    """Cache sampled trajectories and terminal target metadata."""

