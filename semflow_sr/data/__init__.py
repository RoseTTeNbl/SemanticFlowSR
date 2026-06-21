"""Lazy data-package exports.

External baseline environments do not always include torch. Keep package import
lightweight so ``semflow_sr.data.benchmark_loader`` can be used without loading
synthetic training utilities.
"""
from __future__ import annotations


_EXPORTS = {
    "GenConfig": ("synthetic_generator", "GenConfig"),
    "generate_expression": ("synthetic_generator", "generate_expression"),
    "generate_trace_task": ("synthetic_generator", "generate_trace_task"),
    "sample_probe_xy": ("synthetic_generator", "sample_probe_xy"),
    "VelocityTraceDataset": ("trace_dataset", "VelocityTraceDataset"),
    "StepRecord": ("trace_dataset", "StepRecord"),
    "build_step_records": ("trace_dataset", "build_step_records"),
    "collate_velocity": ("collate", "collate_velocity"),
    "SRTask": ("benchmark_loader", "SRTask"),
    "materialize_formula": ("benchmark_loader", "materialize_formula"),
    "PMLBLoader": ("benchmark_loader", "PMLBLoader"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
