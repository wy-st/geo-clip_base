"""
GWF-U — WLS with BNN-style Reparameterized β

The local regression coefficient β is treated as a Bayesian neural-network
weight: it has a deterministic posterior mean (the WLS MAP estimate μ_β) and
a learnable noise scale σ.  During training a single reparameterized sample
is drawn and the model is trained with plain MSE:

    β_sample = μ_β + σ · ε,   ε ~ N(0, I_e),   σ = exp(log_sigma)
    ŷ_sample = z̃_q · β_sample
    L        = MSE(ŷ_sample, y_true)

At inference the MAP prediction (ε = 0) is returned as the point estimate.
Predictive uncertainty is estimated by Monte-Carlo sampling in GWF_U.

The only extra parameter over plain GWF is log_sigma (one scalar).
"""

import torch
import torch.nn as nn


class UncertainMatrixGWR(nn.Module):
    """
    WLS in K_z-projected TabPFN space with BNN reparameterization for β.

    Parameters
    ----------
    lam : float   ridge regularisation  (same as GWF)

    Learnable parameters
    --------------------
    log_sigma : scalar  — log of the β noise scale σ

    Forward
    -------
    eps_beta = None  →  deterministic MAP  β = μ_β
    eps_beta given   →  BNN sample         β = μ_β + exp(log_sigma) · ε

    Returns
    -------
    y_hat : (B,)   — point prediction
    beta  : (B, e) — MAP coefficients μ_β  (always the posterior mean,
                     regardless of eps_beta, for spatial analysis)
    """

    def __init__(self, lam: float = 1e-3):
        super().__init__()
        self.lam = lam
        # Learnable β noise scale; initialised to 0 → σ = 1 at the start
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def forward(self,
                z_query:  torch.Tensor,
                K_z:      torch.Tensor,
                z_nbr:    torch.Tensor,
                y_nbr:    torch.Tensor,
                w:        torch.Tensor,
                eps_beta: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z_query  : (B, tabpfn_dim)
        K_z      : (B, tabpfn_dim, e)
        z_nbr    : (B, k, tabpfn_dim)
        y_nbr    : (B, k)
        w        : (B, k)   spatial attention weights
        eps_beta : (B, e) | None
        """
        B, k, _ = z_nbr.shape
        e = K_z.shape[-1]

        # ── 1. Project into regression subspace ───────────────────────────
        z_q_tilde = torch.bmm(z_query.unsqueeze(1), K_z).squeeze(1)  # (B, e)
        Z_tilde   = torch.bmm(z_nbr, K_z)                             # (B, k, e)

        # ── 2. Normal matrix  A = Z̃ᵀ diag(w) Z̃ + λI ─────────────────────
        w_sqrt = w.sqrt().unsqueeze(-1)
        Zw     = Z_tilde * w_sqrt                                      # (B, k, e)
        A      = torch.bmm(Zw.transpose(1, 2), Zw)                    # (B, e, e)
        A      = A + self.lam * torch.eye(e, device=A.device,
                                          dtype=A.dtype).unsqueeze(0)

        # ── 3. RHS  b = Z̃ᵀ diag(w) y ─────────────────────────────────────
        b = torch.bmm(Z_tilde.transpose(1, 2),
                      (w * y_nbr).unsqueeze(-1)).squeeze(-1)           # (B, e)

        # ── 4. MAP estimate  μ_β = A⁻¹ b ──────────────────────────────────
        mu_beta = torch.linalg.solve(A, b)                             # (B, e)

        # ── 5. BNN reparameterization ──────────────────────────────────────
        #   β = μ_β + σ · ε,   σ = exp(log_sigma)
        if eps_beta is not None:
            beta = mu_beta + self.log_sigma.exp() * eps_beta
        else:
            beta = mu_beta

        # ── 6. Prediction ─────────────────────────────────────────────────
        y_hat = (z_q_tilde * beta).sum(dim=-1)                        # (B,)

        return y_hat, mu_beta
