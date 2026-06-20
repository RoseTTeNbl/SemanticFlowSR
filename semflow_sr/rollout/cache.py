"""Compatibility JSONL cache helpers for legacy rollout/search target records."""
from __future__ import annotations

from pathlib import Path

from ..utils.jsonl_cache import JsonlRecordCache


class RolloutTargetCache(JsonlRecordCache):
    """Legacy name; new code should prefer candidates.CandidateTargetCache."""

    def __init__(self, path: str | Path):
        super().__init__(path)
