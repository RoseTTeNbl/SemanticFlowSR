import torch

from semflow_sr.actions.action_space import ActionSpace, ActionSpec
from semflow_sr.actions.action_executor import ActionExecutor
from semflow_sr.endpoints.target_rollout_fitness import RolloutEvaluator
from semflow_sr.models.velocity_model import lograte_to_velocity_output
from semflow_sr.registers.state import init_register_state
from semflow_sr.search.rollout_velocity import rollout_velocity
from semflow_sr.semantics.energy import ActionEnergyConfig
from semflow_sr.sr.ops import NAME_TO_ID


class _ZeroLograteModel:
    def eval(self):
        return self

    def __call__(self, *, p_lambda, action_mask=None, **kwargs):
        return lograte_to_velocity_output(torch.zeros_like(p_lambda), p_lambda, mask=action_mask)


def test_gp_guided_rollout_policy_prefers_gp_scored_completion_action():
    ops = [NAME_TO_ID["add"], NAME_TO_ID["square"]]
    space = ActionSpace(K=4, allowed_ops=ops)
    state = init_register_state(num_vars=1, K=4)
    x = torch.linspace(-1.0, 1.0, 32).unsqueeze(1)
    y = x.squeeze(1) ** 4
    first_action = int(space.encode(ActionSpec(NAME_TO_ID["add"], 0, 0, 2)))
    after_first = ActionExecutor(space).execute_symbolic(state, first_action)
    valid_after = space.valid_actions(after_first)
    square_xx = int(space.encode(ActionSpec(NAME_TO_ID["square"], 2, 0, 3)))
    assert bool((valid_after == square_xx).any())

    evaluator = RolloutEvaluator(
        space,
        ActionEnergyConfig(lambda_op=0.0),
        max_completion_steps=1,
        rollout_policy="gp_guided",
        gp_action_scores={square_xx: 10.0},
        seed=0,
    )

    result = evaluator.evaluate_after_action(state, first_action, x, y, n_rollouts=1)

    assert result.scores[0].sequence[-1] == square_xx


def test_gp_guided_rollout_policy_accepts_operator_prior_scores():
    ops = [NAME_TO_ID["add"], NAME_TO_ID["square"]]
    space = ActionSpace(K=4, allowed_ops=ops)
    state = init_register_state(num_vars=1, K=4)
    x = torch.linspace(-1.0, 1.0, 32).unsqueeze(1)
    y = x.squeeze(1) ** 4
    first_action = int(space.encode(ActionSpec(NAME_TO_ID["add"], 0, 0, 2)))

    evaluator = RolloutEvaluator(
        space,
        ActionEnergyConfig(lambda_op=0.0),
        max_completion_steps=1,
        rollout_policy="gp_guided",
        gp_operator_scores={"square": 10.0},
        seed=0,
    )

    result = evaluator.evaluate_after_action(state, first_action, x, y, n_rollouts=1)

    assert space.decode(result.scores[0].sequence[-1]).op_id == NAME_TO_ID["square"]


def test_gp_prior_guides_online_policy_update_when_enabled():
    ops = [NAME_TO_ID["add"], NAME_TO_ID["square"]]
    x = torch.linspace(-1.0, 1.0, 16).unsqueeze(1)
    y = x.squeeze(1) ** 2

    baseline = rollout_velocity(
        _ZeroLograteModel(),
        x,
        y,
        num_vars=1,
        K=3,
        ops_ids=ops,
        device=torch.device("cpu"),
        max_steps=1,
        max_support=64,
        support_mode="full",
        greedy=True,
        energy_cfg=ActionEnergyConfig(lambda_op=0.0),
        record_diagnostics=True,
    )
    guided = rollout_velocity(
        _ZeroLograteModel(),
        x,
        y,
        num_vars=1,
        K=3,
        ops_ids=ops,
        device=torch.device("cpu"),
        max_steps=1,
        max_support=64,
        support_mode="full",
        greedy=True,
        energy_cfg=ActionEnergyConfig(lambda_op=0.0),
        gp_operator_scores={"square": 5.0},
        gp_policy_weight=2.0,
        record_diagnostics=True,
    )

    assert baseline.diagnostics[0]["selected_action"]["op"] != "square"
    assert guided.diagnostics[0]["selected_action"]["op"] == "square"
    assert guided.diagnostics[0]["gp_policy_applied"] is True
