#!/usr/bin/env python
"""Generate the local velocity-flow trace dataset and save velocity-matching samples.

Saves a single torch file containing the materialized batches (B,y,actions,energies,
weights,p0,p1,lambda,p_lambda,dp_dlambda,gt_action). Intended for the diagnostic /
overfit experiments; large-scale training can stream from the generator instead.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from semflow_sr.data.synthetic_generator import GenConfig
from semflow_sr.train.build_dataset import build_dataset
from semflow_sr.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_tasks", type=int, default=2000)
    ap.add_argument("--suite", default="mixed")
    ap.add_argument("--num_vars", type=int, default=1)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--probe_size", type=int, default=128)
    ap.add_argument("--target", default="gt", choices=["gt", "semantic_oracle"])
    ap.add_argument("--max_support", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/local_flow_traces/v0")
    a = ap.parse_args()
    set_seed(a.seed)

    gen = GenConfig(num_vars=a.num_vars, max_depth=a.max_depth, K=a.K, probe_size=a.probe_size)
    ds = build_dataset(gen, a.num_tasks, target=a.target, seed=a.seed, max_support=a.max_support)
    samples = [ds[i] for i in range(len(ds))]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples, "config": vars(a)}, out / "traces.pt")
    print(f"saved {len(samples)} step-samples to {out/'traces.pt'}")


if __name__ == "__main__":
    main()
