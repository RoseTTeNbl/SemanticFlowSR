"""Train Edge-Parameterized Semantic Flow on synthetic tasks."""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from ..data.synthetic_generator import GenConfig, generate_expression, sample_probe_xy
from ..sr.printer import to_string
from .benchmark import load_edge_flow_benchmark_tasks, task_tensors
from .dataset import EdgeFlowBuildConfig, build_edge_flow_records
from .model import EdgeFlowModel, EdgeFlowModelConfig, edge_flow_loss
from .template import RegisterOperatorTemplate


def run(cfg: dict) -> Path:
    _configure_threads(cfg.get("runtime", {}))
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    template = _template_from_cfg(cfg)
    model = EdgeFlowModel(EdgeFlowModelConfig(
        num_vars=template.num_vars,
        hidden=int(cfg.get("model", {}).get("hidden", 64)),
    ))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("train", {}).get("lr", 1e-3)))
    rows: list[dict] = []
    train_cfg = cfg.get("train", {})
    build_cfg = EdgeFlowBuildConfig(
        samples_per_task=int(train_cfg.get("samples_per_task", 64)),
        elite_k=int(train_cfg.get("elite_k", 8)),
        target_smoothing=float(train_cfg.get("target_smoothing", 1e-2)),
        complexity_weight=float(train_cfg.get("complexity_weight", 0.001)),
        projection_mode=str(train_cfg.get("projection_mode", "global_topk")),
        validation_fraction=float(train_cfg.get("validation_fraction", 0.0)),
    )
    tasks = _training_tasks(cfg, template, rng)
    epochs = int(train_cfg.get("epochs", 2))
    for epoch in range(epochs):
        records = build_edge_flow_records(template, tasks=tasks, cfg=build_cfg, rng=rng)
        for rec in records:
            pred = model(rec)
            loss, metrics = edge_flow_loss(pred, rec)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            row = {
                "epoch": epoch,
                "task_id": rec.task_id,
                **metrics,
                **{k: v for k, v in rec.diagnostics.items() if isinstance(v, (int, float))},
            }
            rows.append(row)
            print(f"epoch {epoch} task {rec.task_id} loss {metrics['loss']:.6f}")
    out_dir = Path(cfg.get("out", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", "edge_flow_smoke.pt"))
    path = out_dir / ckpt_name
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg,
        "model_cfg": asdict(model.cfg),
        "template": {
            "num_vars": template.num_vars,
            "num_registers": template.num_registers,
            "num_layers": template.num_layers,
            "primitives": list(template.primitives),
            "mixture_modes": template.mixture_modes,
        },
        "algorithm": "edge_parameterized_semantic_flow_matching",
    }, path)
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {path}")
    return path


def _template_from_cfg(cfg: dict) -> RegisterOperatorTemplate:
    t = cfg.get("template", {})
    return RegisterOperatorTemplate(
        num_vars=int(t.get("num_vars", cfg.get("gen", {}).get("num_vars", 1))),
        num_registers=int(t.get("num_registers", 4)),
        num_layers=int(t.get("num_layers", 2)),
        primitives=tuple(t.get("primitives", ["add", "mul", "square"])),
        mixture_modes=int(t.get("mixture_modes", 1)),
    )


def _synthetic_tasks(cfg: dict, rng: random.Random) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    gen_cfg = cfg.get("gen", {})
    template_cfg = cfg.get("template", {})
    gen = GenConfig(
        num_vars=int(template_cfg.get("num_vars", gen_cfg.get("num_vars", 1))),
        max_depth=int(gen_cfg.get("max_depth", 3)),
        K=int(template_cfg.get("num_registers", gen_cfg.get("K", 4))),
        probe_size=int(gen_cfg.get("probe_size", 32)),
        ops=tuple(template_cfg.get("primitives", gen_cfg.get("ops", ["add", "mul", "square"]))),
    )
    tasks = []
    for idx in range(int(cfg.get("num_tasks", 2))):
        expr = generate_expression(gen, rng)
        x, y = sample_probe_xy(expr, gen, rng)
        tasks.append((f"synthetic_{idx}:{to_string(expr, gen.num_vars, simplify=True)}", x, y))
    return tasks


def _training_tasks(
    cfg: dict,
    template: RegisterOperatorTemplate,
    rng: random.Random,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    data_cfg = cfg.get("data", {})
    source = str(data_cfg.get("source", "synthetic")).strip().lower()
    if source in {"synthetic", "generated"}:
        return _synthetic_tasks(cfg, rng)
    if source in {"benchmark", "benchmark_87", "87task", "87_task"}:
        tasks = load_edge_flow_benchmark_tasks(
            manifest=data_cfg.get("manifest", "data/benchmark_suites/benchmark_manifest.json"),
            suites=list(data_cfg.get("suites", ["nguyen", "constant", "livermore", "jin"])),
            root=data_cfg.get("manifest_root", "data/benchmark_suites"),
            seed=int(data_cfg.get("seed", cfg.get("seed", 0))),
            legacy_87=bool(data_cfg.get("legacy_87", True)),
            feynman_root=data_cfg.get("feynman_root", "data/materialized/feynman"),
            limit=data_cfg.get("limit_tasks"),
        )
        out: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for task in tasks:
            if int(task.X_train.shape[1]) > template.num_vars:
                continue
            x_train, y_train, _, _ = task_tensors(task, template_num_vars=template.num_vars)
            out.append((task.name, x_train, y_train))
        if not out:
            raise ValueError("benchmark data source produced no compatible tasks")
        return out
    raise ValueError(f"unknown edge-flow training data source: {source}")


def _write_curve(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _configure_threads(runtime: dict) -> None:
    if runtime.get("torch_num_threads") is not None:
        torch.set_num_threads(max(int(runtime["torch_num_threads"]), 1))
    if runtime.get("torch_num_interop_threads") is not None:
        try:
            torch.set_num_interop_threads(max(int(runtime["torch_num_interop_threads"]), 1))
        except RuntimeError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
