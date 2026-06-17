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
    return clamp_finite(B.transpose(-1, -2) @ B)            # [...,K,K]; guard overflow


def _center_B(B: torch.Tensor) -> torch.Tensor:
    return B - B.mean(dim=-2, keepdim=True)


def _ridge_inv(G: torch.Tensor, rho: float) -> torch.Tensor:
    K = G.shape[-1]
    I = torch.eye(K, device=G.device, dtype=G.dtype).expand_as(G)
    try:
        return torch.linalg.solve(G + rho * I, I)
    except Exception:
        # singular (huge/collinear columns): scale ridge by Gram magnitude, then pinv
        scale = torch.diagonal(G, dim1=-2, dim2=-1).abs().mean(-1, keepdim=True).clamp(min=1.0)
        reg = (rho * scale).unsqueeze(-1) * I
        return torch.linalg.pinv(G + reg)


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
        """中心化岭投影: ŷ = ȳ + Π_{B̃}(y−ȳ), B̃/ỹ 为去均值列/目标。
        截距由去均值隐式吸收, 故常数列无法靠均值偏移获得虚假收益。"""
        Bc = _center_B(B)                                # 列去均值
        yc = y - y.mean(dim=-1, keepdim=True)
        Minv = self._Minv(Bc)
        By = (Bc.transpose(-1, -2) @ yc.unsqueeze(-1))    # [...,K,1]
        coeff = Minv @ By
        fit = (Bc @ coeff).squeeze(-1) + y.mean(dim=-1, keepdim=True)
        return clamp_finite(fit)

    def residual_energy(self, B: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = self.residual_vector(B, y)
        return 0.5 * (r * r).sum(dim=-1)

    def residual_vector(self, B: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Centered residual vector ``y - Π_B y`` on the shared projection backend."""
        return clamp_finite(y - self.project_y(B, y))

    def explained_energy(self, B: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        py = self.project_y(B, y)
        return 0.5 * (py * py).sum(dim=-1)

    def effective_rank(self, B: torch.Tensor) -> torch.Tensor:
        """Tr(Π_{B,ρ}) = Tr((G+ρI)^{-1} G)."""
        Bc = _center_B(B)
        G = _gram(Bc)
        Minv = self._Minv(Bc)
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
        Bc = _center_B(B)
        G = _gram(Bc)
        M = self._Minv(Bc)
        prod = M @ G @ M @ G
        return torch.diagonal(prod, dim1=-2, dim2=-1).sum(dim=-1)

    def _trace_cross(self, B1: torch.Tensor, B2: torch.Tensor) -> torch.Tensor:
        """Tr(Π1 Π2) with Πi = Bi Mi Biᵀ, Mi=(Gi+ρI)^{-1}.
        = Tr( M1 (B1ᵀB2) M2 (B2ᵀB1) )."""
        B1c = _center_B(B1)
        B2c = _center_B(B2)
        M1 = self._Minv(B1c)
        M2 = self._Minv(B2c)
        C12 = B1c.transpose(-1, -2) @ B2c          # [...,K,K]
        C21 = C12.transpose(-1, -2)
        prod = M1 @ C12 @ M2 @ C21
        return torch.diagonal(prod, dim1=-2, dim2=-1).sum(dim=-1)
