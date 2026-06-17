from .metrics import r2_score, nmse, accuracy_rate, energy_decrease_ratio


def __getattr__(name):  # lazy: keep evaluator (torch) out of import path for torch-free baselines
    if name in ("evaluate_task", "EvalReport"):
        from . import evaluator
        return getattr(evaluator, name)
    raise AttributeError(name)
