import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "distill_gp_events_from_results.py"
    spec = importlib.util.spec_from_file_location("distill_gp_events_from_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


expression_operator_events = _load_script().expression_operator_events


def test_expression_operator_events_extracts_operator_likelihood_events():
    events = expression_operator_events(
        "toy",
        {"r2": 1.0, "expression": "add(mul(ARG0, ARG0), sin(ARG1))"},
    )

    ops = {event["op"] for event in events}
    assert {"add", "mul", "sin"} <= ops
    assert all(event["solved"] for event in events)
    assert all(event["r2"] == 1.0 for event in events)
