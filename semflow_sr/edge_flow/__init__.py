"""Semantic Pullback Fisher Flow package.

The benchmark helpers in this package are used by paper-table builders and must
remain importable even when historical SPFF training-only modules are absent.
Training symbols are therefore imported lazily.
"""

from importlib import import_module

__all__ = [
    "ConditionalEdgeFlowConfig",
    "ConditionalEdgeFlowModel",
    "ConditionalEdgeFlowSampler",
    "conditional_elite_policy_loss",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(".conditional", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
