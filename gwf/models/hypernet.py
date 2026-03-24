"""
gwf/models/hypernet.py
======================
Module 6: HyperNetwork — Beta Coefficient Generation

Generates spatially-varying, high-dimensional regression coefficients Beta_i
for each spatial point. This replaces the matrix inversion (X^T W X)^{-1} X^T W Y
of classical GWR with a learned HyperNetwork.

Beta is stored in LOW-RANK form:
    Beta_i = U_i @ diag(sigma_i) @ V_i^T
    U_i    : (d, r)   left singular vectors
    sigma_i: (r,)     singular values
    V_i    : (r, d)   right singular vectors
    r << d, default r=32, d=512

This gives O(dr) storage per point instead of O(d²). Memory efficient AND
interpretable: sigma_i captures "how strongly" each component contributes.

Two modes:
  V1 (probabilistic=False): deterministic, directly output U, sigma, V
  V2 (probabilistic=True) : variational, sample U, sigma, V from Gaussian,
                            also returns KL divergence for regularisation

Input to HyperNet:
    h_spatial (N, d)  — spatially contextualised embeddings from GNN
    c_task    (N, d)  — task context from LLM bridge

Output:
    Beta dict: {"U": (N,d,r), "sigma": (N,r), "V": (N,r,d)}
    V2 also adds: {"kl_loss": scalar}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNetBeta(nn.Module):
    """
    HyperNetwork that generates per-point low-rank Beta coefficients.

    Args:
        d             : embedding dimension (default 512)
        r             : Beta rank (default 32)
        probabilistic : if True → V2 variational mode, else V1 deterministic
    """

    def __init__(self, d: int = 512, r: int = 32, probabilistic: bool = False):
        super().__init__()
        self.d = d
        self.r = r
        self.probabilistic = probabilistic

        # Shared trunk: [h_spatial || c_task] (2d) → (d)
        self.shared = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        if not probabilistic:
            # ===== V1: DETERMINISTIC =====
            self.head_U     = nn.Linear(d, d * r)       # → (N, d*r)
            self.head_sigma = nn.Linear(d, r)            # → (N, r)
            self.head_V     = nn.Linear(d, r * d)       # → (N, r*d)

        else:
            # ===== V2: PROBABILISTIC (variational) =====
            # U: mean + log-variance
            self.head_U_mean   = nn.Linear(d, d * r)
            self.head_U_logvar = nn.Linear(d, d * r)
            # sigma: mean + log-variance
            self.head_sigma_mean   = nn.Linear(d, r)
            self.head_sigma_logvar = nn.Linear(d, r)
            # V: mean + log-variance
            self.head_V_mean   = nn.Linear(d, r * d)
            self.head_V_logvar = nn.Linear(d, r * d)

    # -------------------------------------------------------------------------
    # Internal: reparameterisation trick
    # -------------------------------------------------------------------------

    @staticmethod
    def _reparameterise(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mean + eps * std, eps ~ N(0, I)."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mean + eps * std

    @staticmethod
    def _kl_gaussian(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL(q || N(0,I)) = -0.5 * sum(1 + logvar - mean² - exp(logvar))
        Returns a scalar (summed over all elements, averaged over batch).
        """
        N = mean.shape[0]
        kl = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp())
        return kl.sum() / N

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        h_spatial: torch.Tensor,          # (N, d)
        c_task:    torch.Tensor,          # (N, d)
        deterministic: bool = False,       # if True and probabilistic: use means
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            Beta dict with keys "U", "sigma", "V" (and "kl_loss" if V2).
        """
        N = h_spatial.shape[0]
        d, r = self.d, self.r

        # ---- Shared trunk ----
        h_cat  = torch.cat([h_spatial, c_task], dim=-1)   # (N, 2d)
        shared = self.shared(h_cat)                         # (N, d)

        if not self.probabilistic:
            # ===== V1: Deterministic =====
            U     = self.head_U(shared).view(N, d, r)       # (N, d, r)
            sigma = self.head_sigma(shared)                   # (N, r)
            V     = self.head_V(shared).view(N, r, d)       # (N, r, d)
            return {"U": U, "sigma": sigma, "V": V}

        else:
            # ===== V2: Variational =====
            # U
            U_mean   = self.head_U_mean(shared).view(N, d, r)
            U_logvar = self.head_U_logvar(shared).view(N, d, r)
            # sigma
            s_mean   = self.head_sigma_mean(shared)
            s_logvar = self.head_sigma_logvar(shared)
            # V
            V_mean   = self.head_V_mean(shared).view(N, r, d)
            V_logvar = self.head_V_logvar(shared).view(N, r, d)

            if deterministic or not self.training:
                # During inference: use posterior means directly
                U, sigma, V = U_mean, s_mean, V_mean
                kl = torch.tensor(0.0, device=h_spatial.device)
            else:
                # During training: reparameterise + compute KL
                U     = self._reparameterise(U_mean,  U_logvar)
                sigma = self._reparameterise(s_mean,  s_logvar)
                V     = self._reparameterise(V_mean,  V_logvar)
                kl = (
                    self._kl_gaussian(U_mean,  U_logvar)
                    + self._kl_gaussian(s_mean, s_logvar)
                    + self._kl_gaussian(V_mean, V_logvar)
                )

            return {"U": U, "sigma": sigma, "V": V, "kl_loss": kl}
