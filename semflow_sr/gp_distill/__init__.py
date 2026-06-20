"""Independent GP implicit-distribution distillation extensions."""

from .kl_barycenter import kl_barycenter
from .trace_likelihood import compute_gp_individual_logprob
from .trajectory_pool import load_gp_trajectory_population

__all__ = ["compute_gp_individual_logprob", "kl_barycenter", "load_gp_trajectory_population"]
