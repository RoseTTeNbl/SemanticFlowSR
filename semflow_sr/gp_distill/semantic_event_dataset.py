"""Lightweight dataset reader for GP semantic events."""
from __future__ import annotations

from pathlib import Path
import json


class GPSemanticEventDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records = []
        if self.path.exists():
            with self.path.open() as f:
                self.records = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]
