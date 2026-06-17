import random
import torch
from semflow_sr.data.synthetic_generator import GenConfig
from semflow_sr.train.build_dataset import build_dataset
from semflow_sr.data.collate import collate_velocity
from semflow_sr.flow.semantic_fisher import semantic_fisher_sphere_step
from semflow_sr.models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig
from semflow_sr.train.trainer_velocity import train_velocity, TrainConfig


def test_overfit_tiny_semantic_fisher_dataset():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    gen = GenConfig(num_vars=1, max_depth=3, K=6, probe_size=32,
                    ops=("mul", "protected_div", "sin", "cos", "square"))
    ds = build_dataset(gen, num_tasks=8, target="gt", seed=0, max_support=64)
    assert len(ds) > 0
    model = SemanticTransformer(SemanticTransformerConfig(
        d=1, K=6, hidden=32, row_layers=1, heads=2, output_mode="semantic_fisher_lograte"
    ))
    cfg = TrainConfig(lr=1e-3, steps=180, batch_size=4, log_every=1000)
    stats = train_velocity(model, ds, cfg, torch.device("cpu"), collate_velocity)
    h = stats["loss_history"]
    # Semantic-Fisher tangent loss should drop substantially on the tiny trace set.
    assert h[-1] < 0.5 * (sum(h[:5]) / 5)


def test_default_dataset_emits_semantic_fisher_targets():
    torch.manual_seed(3)
    gen = GenConfig(num_vars=1, max_depth=3, K=6, probe_size=24,
                    ops=("add", "sub", "mul", "square"))
    ds = build_dataset(gen, num_tasks=4, target="group_advantage", seed=3, max_support=24)
    sample = ds[0]

    assert sample["lambda"].item() == 0.0
    assert sample["gram"].shape == (sample["action_ids"].numel(), sample["action_ids"].numel())
    assert sample["xi"].shape == (sample["action_ids"].numel(), sample["x"].shape[0])
    assert sample["semantic_stats"].shape == (sample["action_ids"].numel(), 8)
    assert torch.allclose(sample["p_lambda"], sample["p_start"])
    assert torch.allclose(sample["dz_dlambda"], sample["zdot_target"])
    assert torch.allclose(
        sample["p_target"],
        semantic_fisher_sphere_step(sample["p_start"], sample["w_target"], dt=1.0),
        atol=1e-12,
    )


def test_trace_dataset_caches_static_support_but_resamples_lambda():
    torch.manual_seed(1)
    gen = GenConfig(num_vars=1, max_depth=3, K=6, probe_size=32,
                    ops=("add", "sub", "mul", "square"))
    ds = build_dataset(
        gen, num_tasks=4, target="group_advantage", seed=1, max_support=32,
        path_name="exponential_natural_flow",
    )
    assert len(ds) > 0

    first = ds[0]
    second = ds[0]

    assert len(ds._static_cache) == 1
    for key in [
        "action_ids", "rewards", "advantages", "proposal_probs",
        "weights", "p_start", "p_target", "p0", "p1",
    ]:
        assert torch.equal(first[key], second[key]), key
    assert torch.equal(first["p_start"], first["p0"])
    assert torch.equal(first["p_target"], first["p1"])
    assert torch.allclose(first["p_target"], torch.softmax(first["p_start"].log() + ds.eta * first["advantages"], dim=-1))
    assert first["lambda"] != second["lambda"]


def test_train_velocity_does_not_eval_at_step_zero():
    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return {
                "x": torch.zeros(4, 1),
                "y": torch.zeros(4),
                "B": torch.zeros(4, 3),
                "action_ids": torch.arange(2),
                "action_feats": torch.zeros(2, 17),
                "semantic_stats": torch.zeros(2, 8),
                "action_mask": torch.ones(2, dtype=torch.bool),
                "energies": torch.zeros(2),
                "weights": torch.ones(2),
                "gram": torch.eye(2),
                "w_target": torch.zeros(2),
                "pdot_target": torch.zeros(2),
                "zdot_target": torch.zeros(2),
                "p_lambda": torch.full((2,), 0.5),
                "dp_dlambda": torch.zeros(2),
                "lambda": torch.tensor(0.5),
                "p_start": torch.full((2,), 0.5),
                "p_target": torch.full((2,), 0.5),
                "plain_p_target": torch.full((2,), 0.5),
                "p0": torch.full((2,), 0.5),
                "p1": torch.full((2,), 0.5),
                "rewards": torch.zeros(2),
                "advantages": torch.zeros(2),
                "proposal_probs": torch.ones(2),
                "gamma": torch.tensor(0.1),
                "gt_action_pos": torch.tensor(-1),
                "full_action_size": torch.tensor(2),
            }

    calls = {"n": 0}
    def eval_fn(_model):
        calls["n"] += 1
        return 0.0

    model = SemanticTransformer(SemanticTransformerConfig(
        d=1, K=3, hidden=16, row_layers=1, heads=1, output_mode="semantic_fisher_lograte"
    ))
    cfg = TrainConfig(lr=1e-3, steps=1, batch_size=1, log_every=1)
    train_velocity(model, TinyDataset(), cfg, torch.device("cpu"), collate_velocity,
                   eval_fn=eval_fn, eval_every=1)
    assert calls["n"] == 0
