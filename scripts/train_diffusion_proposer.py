#!/usr/bin/env python
"""Train an Approach3-style text diffusion proposer and export formula proposals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from semflow_sr.edge_flow.diffusion_proposer import (
    generate_proposals_jsonl,
    load_diffusion_checkpoint,
    load_symbolicgpt_records,
    save_diffusion_checkpoint,
    train_diffusion_proposer,
)


def run_from_args(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/generated/symbolicgpt_subset")
    parser.add_argument("--ckpt", default="checkpoints/diffusion_proposer_symbolicgpt.pt")
    parser.add_argument("--proposals_out", default="results/diffusion_proposer_symbolicgpt/proposals_train.jsonl")
    parser.add_argument("--curve_out", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--proposal_splits", nargs="+", default=["train"])
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_vars", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proposals_per_task", type=int, default=8)
    parser.add_argument("--generation_mode", default="teacher_noised", choices=["teacher_noised", "teacher", "approach3", "random"])
    parser.add_argument("--t_value", type=int, default=None)
    parser.add_argument("--fallback_to_teacher", action="store_true")
    parser.add_argument("--generate_only", action="store_true")
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    if bool(args.generate_only):
        model, metadata = load_diffusion_checkpoint(args.ckpt, device=device)
        metadata = dict(metadata)
        train_records = load_symbolicgpt_records(
            args.root,
            splits=(str(args.train_split),),
            limit=_optional_limit(args.train_limit),
        )
        val_records = load_symbolicgpt_records(
            args.root,
            splits=(str(args.val_split),),
            limit=_optional_limit(args.val_limit),
        )
    else:
        train_records = load_symbolicgpt_records(
            args.root,
            splits=(str(args.train_split),),
            limit=_optional_limit(args.train_limit),
        )
        val_records = load_symbolicgpt_records(
            args.root,
            splits=(str(args.val_split),),
            limit=_optional_limit(args.val_limit),
        )
        model, metadata = train_diffusion_proposer(
            train_records,
            val_records,
            num_vars=int(args.num_vars),
            hidden=int(args.hidden),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            patience=int(args.patience),
            device=device,
            seed=int(args.seed),
        )
        metadata = dict(metadata)
        save_diffusion_checkpoint(args.ckpt, model, metadata)
    metadata["train_records"] = int(len(train_records))
    metadata["val_records"] = int(len(val_records))
    metadata["generation_mode"] = str(args.generation_mode)
    metadata["fallback_to_teacher"] = bool(args.fallback_to_teacher)

    proposal_records = load_symbolicgpt_records(
        args.root,
        splits=tuple(str(item) for item in args.proposal_splits),
        limit=None,
    )
    proposal_rows = generate_proposals_jsonl(
        proposal_records,
        model=model,
        metadata=metadata,
        out=args.proposals_out,
        device=device,
        proposals_per_task=int(args.proposals_per_task),
        mode=str(args.generation_mode),
        t_value=args.t_value,
        fallback_to_teacher=bool(args.fallback_to_teacher),
        seed=int(args.seed),
    )
    curve_out = Path(args.curve_out) if str(args.curve_out).strip() else Path(args.ckpt).with_suffix(".curve.json")
    curve_out.parent.mkdir(parents=True, exist_ok=True)
    curve_out.write_text(json.dumps({
        "checkpoint": str(args.ckpt),
        "proposals_out": str(args.proposals_out),
        "proposal_rows": int(proposal_rows),
        "metadata": _jsonable_metadata(metadata),
    }, indent=2))
    summary = {
        "checkpoint": str(args.ckpt),
        "curve": str(curve_out),
        "proposals_out": str(args.proposals_out),
        "train_records": int(len(train_records)),
        "val_records": int(len(val_records)),
        "proposal_rows": int(proposal_rows),
        "best_val_loss": float(metadata.get("best_val_loss", 0.0)),
        "device": str(device),
    }
    print(json.dumps(summary, indent=2))
    return summary


def _optional_limit(value: int) -> int | None:
    value = int(value)
    return None if value <= 0 else value


def _resolve_device(text: str) -> torch.device:
    requested = str(text).strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested cuda, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
        index_text = requested.split(":", 1)[1]
        if not index_text.isdigit():
            raise ValueError(f"invalid CUDA device specifier: {requested}")
        index = int(index_text)
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"requested {requested}, but only {torch.cuda.device_count()} CUDA devices are visible")
        return torch.device(requested)
    raise ValueError(f"unsupported device: {text}")


def _jsonable_metadata(metadata: dict) -> dict:
    out = dict(metadata)
    if "curve" in out:
        out["curve"] = [dict(row) for row in out["curve"]]
    if "vocab" in out:
        out["vocab"] = {str(k): int(v) for k, v in dict(out["vocab"]).items()}
    return out


def main() -> None:
    run_from_args()


if __name__ == "__main__":
    main()
