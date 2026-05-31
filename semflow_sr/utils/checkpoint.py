"""Checkpoint save/load."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, meta: dict | None = None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict(), "meta": meta or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("meta", {})
