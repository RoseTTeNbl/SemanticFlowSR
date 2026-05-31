"""Projection backends.

Ridge (default):   Π_{B,ρ} = B (BᵀB + ρI)⁻¹ Bᵀ
Hard (ablation):   Π_B    = Q_B Q_Bᵀ  (Q from thin QR)

We never materialize the m×m projector. Key quantities are computed via the K×K Gram
matrix G = BᵀB:
  project_y:      Π y  = B (G+ρI)⁻¹ Bᵀ y
  residual_energy ½‖y-Πy‖²
  explained_energy ½‖Πy‖²
  effective_rank  Tr(Π_{B,ρ}) = Tr((G+ρI)⁻¹ G)
  projection_distance ‖Π1-Π2‖²_F = Tr(Π1)+Tr(Π2)-2 Tr(Π1 Π2)
All batched over a leading action dimension when B is [A,m,K].
"""
from __future__ import annotations
import torch
from ..utils.numerical import clamp_finite


def _gram(B: torch.Tensor) -> torch.Tensor:
    return B.transpose(-1, -2) @ B            # [...,K,K]


def _ridge_inv(G: torch.Tensor, rho: float) -> torch.Tensor:
    K = G.shape[-1]
    I = torch.eye(K, device=G.device, dtype=G.dtype).expand_as(G)
    return torch.linalg.solve(G + rho * I, I)


class ProjectionBackend:
    def __init__(self, mode: str = "ridge", rho: float = 1e-3):
        assert mode in ("ridge", "hard")
        self.mode = mode
        self.rho = rho

    # --- ridge projector matrix in K-space: M = (G+ρI)^{-1} ---
    def _Minv(self, B: torch.Tensor) -> torch.Tensor:
        G = _gram(B)
        if self.mode == "ridge":
            return _ridge_inv(G, self.rho)
        # hard: pseudo-inverse of G (Π = B G^+ Bᵀ)
        return torch.linalg.pinv(G)

    def project_y(self, B: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Π y. B:[...,m,K], y:[m] or [...,m] -> [...,m]."""
        Minv = self._Minv(B)
        By = (B.transpose(-1, -2) @ y.unsqueeze(-1))      # [...,K,1]
        coeff = Minv @ By                                  # [...,K,1]
        return clamp_finite((B @ coeff).squeeze(-1))

    def residual_energy(self, B: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = y - self.project_y(B, y)
        return 0.5 * (r * r).sum(dim=-1)

    def explained_energy(self, B: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        py = self.project_y(B, y)
        return 0.5 * (py * py).sum(dim=-1)

    def effective_rank(self, B: torch.Tensor) -> torch.Tensor:
        """Tr(Π_{B,ρ}) = Tr((G+ρI)^{-1} G)."""
        G = _gram(B)
        Minv = self._Minv(B)
        prod = Minv @ G
        return clamp_finite(torch.diagonal(prod, dim1=-2, dim2=-1).sum(dim=-1))

    def projection_distance(self, B1: torch.Tensor, B2: torch.Tensor) -> torch.Tensor:
        """‖Π1-Π2‖²_F = Tr(Π1²)+Tr(Π2²)-2Tr(Π1Π2).
        Ridge projectors are NOT idempotent, so the diagonal terms use Tr(Πᵢ²), not Tr(Πᵢ).
        B1 may be [A,m,K] (batched), B2 [m,K] -> broadcast. Returns [...]."""
        t1 = self._trace_proj_sq(B1)
        t2 = self._trace_proj_sq(B2)
        cross = self._trace_cross(B1, B2)
        return clamp_finite(t1 + t2 - 2.0 * cross)

    # helpers
    def _trace_proj_sq(self, B: torch.Tensor) -> torch.Tensor:
        """Tr(Π²) = Tr(M G M G) with M=(G+ρI)^{-1}, G=BᵀB."""
        G = _gram(B)
        M = self._Minv(B)
        prod = M @ G @ M @ G
        return torch.diagonal(prod, dim1=-2, dim2=-1).sum(dim=-1)

    def _trace_cross(self, B1: torch.Tensor, B2: torch.Tensor) -> torch.Tensor:
        """Tr(Π1 Π2) with Πi = Bi Mi Biᵀ, Mi=(Gi+ρI)^{-1}.
        = Tr( M1 (B1ᵀB2) M2 (B2ᵀB1) )."""
        M1 = self._Minv(B1)
        M2 = self._Minv(B2)
        C12 = B1.transpose(-1, -2) @ B2            # [...,K,K]
        C21 = C12.transpose(-1, -2)
        prod = M1 @ C12 @ M2 @ C21
        return torch.diagonal(prod, dim1=-2, dim2=-1).sum(dim=-1)
