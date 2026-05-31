import torch
from semflow_sr.data.synthetic_generator import GenConfig
from semflow_sr.train.build_dataset import build_dataset
from semflow_sr.data.collate import collate_velocity
from semflow_sr.models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig
from semflow_sr.train.trainer_velocity import train_velocity, TrainConfig
from semflow_sr.search.rollout_velocity import rollout_velocity, rollout_random
from semflow_sr.sr.ops import NAME_TO_ID
from semflow_sr.sr.parser import parse_formula
from semflow_sr.sr.ast import eval_expr


def test_rollout_beats_random_policy():
    torch.manual_seed(0)
    ops = ("add", "sub", "mul", "sin", "cos", "square")
    ops_ids = [NAME_TO_ID[o] for o in ops]
    gen = GenConfig(num_vars=1, max_depth=3, K=6, probe_size=32, ops=ops)
    ds = build_dataset(gen, num_tasks=60, target="semantic_oracle", seed=1, max_support=64)
    model = SemanticTransformer(SemanticTransformerConfig(d=1, K=6, hidden=32, row_layers=1, heads=2))
    cfg = TrainConfig(lr=1e-3, steps=400, batch_size=8, log_every=10000)
    train_velocity(model, ds, cfg, torch.device("cpu"), collate_velocity)

    # target task
    expr = parse_formula("x0*x0 + x0", ["x0"])
    x = torch.linspace(-1, 1, 64).unsqueeze(1)
    y = eval_expr(expr, x)

    learned, rand = [], []
    for seed in range(5):
        torch.manual_seed(seed)
        r1 = rollout_velocity(model, x, y, 1, 6, ops_ids, torch.device("cpu"),
                              max_steps=8, grid=4, greedy=True)
        r2 = rollout_random(x, y, 1, 6, ops_ids, torch.device("cpu"), max_steps=8, seed=seed)
        learned.append(min(r1.energy_trace))
        rand.append(min(r2.energy_trace))
    # learned policy should reach lower residual energy on average than random
    assert sum(learned) / len(learned) <= sum(rand) / len(rand) + 1e-6
