"""JSONL cache helpers for rollout/search target records."""
from __future__ import annotations

from pathlib import Path
import json


class RolloutTargetCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
