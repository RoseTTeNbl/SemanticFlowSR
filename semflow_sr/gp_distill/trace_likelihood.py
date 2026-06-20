"""Likelihood accounting for GP trace distillation diagnostics."""
from __future__ import annotations

from collections.abc import Iterable

import torch


LOGPROB_KEYS = {
    "logprob",
    "parent_selection_logprob",
    "variation_operator_logprob",
    "crossover_site_logprob",
    "crossover_sites_logprob",
    "mutation_site_logprob",
    "mutation_sites_logprob",
    "mutation_content_logprob",
    "constant_logprob",
    "constants_logprob",
}


def compute_gp_individual_logprob(event_log) -> torch.Tensor:
    """Sum explicit stochastic log-probability terms in a GP event log.

    Deterministic transforms may be present in the log but do not add probability
    mass unless they carry one of the explicit ``*_logprob`` fields.
    """
    if isinstance(event_log, dict):
        events = event_log.get("events", [event_log])
    else:
        events = event_log
    if not isinstance(events, Iterable):
        raise TypeError("event_log must be a dict or iterable of dict events")
    total = 0.0
    for event in events:
        if not isinstance(event, dict):
            continue
        for key, value in event.items():
            if key in LOGPROB_KEYS and value is not None:
                total += float(value)
    return torch.tensor(total, dtype=torch.float32)
