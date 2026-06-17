"""Entry: train a SemanticFlowSR update model.

Usage: python -m semflow_sr.train.train_velocity_gt --config configs/train/velocity_gt.yaml
"""
from __future__ import annotations
import argparse, csv, yaml
from pathlib import Path

from .build_dataset import build_dataset
from .trainer_velocity import train_velocity, TrainConfig
from ..data.synthetic_generator import GenConfig, generate_trace_task
from ..data.collate import collate_velocity
from ..models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig
from ..semantics.energy import ActionEnergyConfig
from ..utils.seed import set_seed, get_device
from ..utils.checkpoint import save_checkpoint


def _configure_runtime(cfg: dict):
    import torch
    runtime = cfg.get("runtime", {})
    if "torch_num_threads" in runtime:
        torch.set_num_threads(int(runtime["torch_num_threads"]))
    if "torch_num_interop_threads" in runtime:
        try:
            torch.set_num_interop_threads(int(runtime["torch_num_interop_threads"]))
        except RuntimeError:
            pass


def _make_eval_fn(gen, ecfg, device, seed, n=8, gamma: float = 0.1, step_dt: float = 1.0):
    """构建 reward 验证: 在 n 个 held-out 合成任务上 rollout 取平均 R², 作为训练 reward 曲线。"""
    import random, torch, numpy as np
    from ..search.rollout_velocity import rollout_velocity
    from ..registers.executor import evaluate_register_state
    from ..sr.ops import NAME_TO_ID
    rng = random.Random(seed + 99999)
    ops_ids = [NAME_TO_ID[o] for o in gen.ops]
    tasks = []
    while len(tasks) < n:
        t = generate_trace_task(gen, rng)
        if t is not None:
            _, _, x, y = t
            tasks.append((x, y))

    def eval_fn(model):
        r2s = []
        for x, y in tasks:
            res = rollout_velocity(model, x, y, gen.num_vars, gen.K, ops_ids, device,
                                   max_steps=16, grid=4, step_size=step_dt,
                                   energy_cfg=ecfg, gamma=gamma)
            B = torch.nan_to_num(evaluate_register_state(res.state, x)).cpu().numpy()
            yy = y.cpu().numpy()
            A = np.nan_to_num(np.concatenate([B, np.ones((B.shape[0], 1))], 1))
            try:
                c = np.linalg.lstsq(A, yy, rcond=None)[0]; pred = A @ c
                ss = ((yy - yy.mean()) ** 2).sum()
                r2s.append(1 - ((yy - pred) ** 2).sum() / ss if ss > 1e-9 else 0.0)
            except Exception:
                r2s.append(0.0)
        return float(np.mean(r2s))
    return eval_fn


def _target_kwargs(cfg: dict) -> dict:
    out = dict(cfg.get("target", {}))
    out.pop("name", None)
    beta = _update_beta(cfg)
    out.setdefault("eta_adv", beta)  # legacy endpoint API; beta is the single main strength.
    rollout_cfg = dict(cfg.get("rollout_target", {}))
    # Offline rollout target data is a future dataset mode; online target construction
    # consumes only the rollout-scoring knobs.
    rollout_cfg.pop("mode", None)
    rollout_cfg.pop("cache_path", None)
    out.update(rollout_cfg)
    return out


def _update_beta(cfg: dict) -> float:
    update = cfg.get("update", {})
    if update.get("mode", "fixed_beta") == "fixed_beta":
        return float(update.get("beta", cfg.get("beta", cfg.get("eta", 1.0))))
    return float(update.get("beta", cfg.get("beta", cfg.get("eta", 1.0))))


def _train_config(cfg: dict) -> TrainConfig:
    train_cfg = dict(cfg.get("train", {}))
    loss_cfg = cfg.get("loss", {})
    path_cfg = cfg.get("path", {})
    update_cfg = cfg.get("update", {})
    train_cfg.setdefault("loss_name", loss_cfg.get("name", "semantic_fisher_velocity"))
    train_cfg.setdefault("lambda_min", loss_cfg.get("lambda_min", path_cfg.get("lambda_min", 0.05)))
    train_cfg.setdefault("num_lambda_samples", loss_cfg.get("num_lambda_samples", 1))
    train_cfg.setdefault("beta", float(update_cfg.get("beta", cfg.get("beta", cfg.get("eta", 1.0)))))
    train_cfg.setdefault("sf_step_dt", float(path_cfg.get("step_dt", 1.0)))
    return TrainConfig(**train_cfg)


def run(cfg: dict, target: str):
    _configure_runtime(cfg)
    set_seed(cfg.get("seed", 0))
    device = get_device()
    g = cfg["gen"]
    gen = GenConfig(num_vars=g["num_vars"], max_depth=g["max_depth"], K=g["K"],
                    probe_size=g["probe_size"], ops=tuple(g["ops"]))
    ecfg = ActionEnergyConfig(**cfg.get("energy", {}))
    beta = _update_beta(cfg)
    ds = build_dataset(gen, cfg["num_tasks"], target=target, beta=beta,
                       seed=cfg.get("seed", 0), max_support=cfg.get("max_support", 256),
                       energy_cfg=ecfg,
                       support_mode=cfg.get("support", {}).get("mode", "mixed_topk_random"),
                       support_topk=cfg.get("support", {}).get("topk"),
                       support_full_threshold=cfg.get("support", {}).get("full_threshold"),
                       target_kwargs=_target_kwargs(cfg),
                       cache_static=cfg.get("data", {}).get("cache_static", True),
                       data_device=cfg.get("data", {}).get("device", "cpu"),
                       path_name=cfg.get("path", {}).get("name", "semantic_fisher_pullback"),
                       gamma=float(cfg.get("path", {}).get("gamma", 0.1)),
                       gram_rank=cfg.get("path", {}).get("gram_rank"),
                       flow_training=cfg.get("flow_training", {}))
    model = SemanticTransformer(SemanticTransformerConfig(
        d=gen.num_vars, K=gen.K, hidden=cfg["model"]["hidden"],
        row_layers=cfg["model"]["row_layers"], heads=cfg["model"]["heads"],
        output_mode=cfg["model"].get("output_mode", "semantic_fisher_lograte")))
    tcfg = _train_config(cfg)
    eval_every = cfg.get("eval_every", max(tcfg.log_every * 5, 200))
    eval_fn = _make_eval_fn(
        gen,
        ecfg,
        device,
        cfg.get("seed", 0),
        gamma=float(cfg.get("path", {}).get("gamma", 0.1)),
        step_dt=float(cfg.get("path", {}).get("step_dt", 1.0)),
    )
    stats = train_velocity(model, ds, tcfg, device, collate_velocity,
                           eval_fn=eval_fn, eval_every=eval_every)
    checkpoint_name = cfg.get("checkpoint_name", f"velocity_{target}.pt")
    out = Path(cfg.get("out", "checkpoints")) / checkpoint_name
    save_checkpoint(out, model, meta={"final_loss": stats["final_loss"], "cfg": cfg})
    _save_curves(stats["log_rows"], Path(cfg.get("out", "checkpoints")), Path(checkpoint_name).stem)
    print(f"saved {out}  final_loss={stats['final_loss']}")


def target_from_config(cfg: dict, default: str = "gt") -> str:
    return str(cfg.get("target", {}).get("name", default))


def _save_curves(rows, out_dir: Path, run_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"train_curve_{run_name}.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else ["step", "epoch", "loss", "reward"]
        for k in ["step", "epoch", "loss", "reward"]:
            if k in fieldnames:
                fieldnames.remove(k)
        fieldnames = ["step", "epoch", "loss", "reward"] + fieldnames
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        steps = [r["step"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(steps, [r["loss"] for r in rows], "b-", label="loss")
        ax1.set_xlabel("step"); ax1.set_ylabel("loss", color="b"); ax1.set_yscale("log")
        rw = [(r["step"], r["reward"]) for r in rows if r["reward"] is not None]
        if rw:
            ax2 = ax1.twinx()
            ax2.plot([s for s, _ in rw], [v for _, v in rw], "r.-", label="reward(val R²)")
            ax2.set_ylabel("reward (val R²)", color="r")
        fig.tight_layout(); fig.savefig(out_dir / f"train_curve_{run_name}.png", dpi=120)
        print(f"saved curve {out_dir / f'train_curve_{run_name}.png'}")
    except Exception as e:
        print(f"plot skipped: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    run(cfg, target=target_from_config(cfg))


if __name__ == "__main__":
    main()
