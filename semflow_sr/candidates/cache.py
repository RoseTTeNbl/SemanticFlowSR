"""Offline cache for candidate trajectory target records."""
from __future__ import annotations

from ..utils.jsonl_cache import JsonlRecordCache


class CandidateTargetCache(JsonlRecordCache):
    """JSONL cache for sampled candidates and their terminal target metadata."""

