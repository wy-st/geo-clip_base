"""
GWF-U — Differentiable WLS with Bayesian Posterior Uncertainty

The ridge WLS solution has a natural Bayesian interpretation:
  Prior   :  β ~ N(0, (1/λ) I)
  Likelihood: y_j | z̃_j, β ~ N(z̃_j · β, σ²)  (i.i.d. within the local window,
                                                   scaled by attention weight w_j)

MAP estimate (= ridge WLS):
  A    = Z̃^T diag(w) Z̃ + λI         (regularised normal matrix)
  β̂   = A^{-1} Z̃^T diag(w) y

Posterior covariance of β (conditioned on σ²):
  Cov(β | σ²) = σ² A^{-1}

Local observation noise σ² estimated from weighted residuals
(no new learnable parameters):
  r_j  = y_j - z̃_j · β̂              (local residuals)
  σ²   = Σ_j w_j r_j²               (weighted mean-square residual)

Prediction variance at query point (posterior predictive):
  Var(ŷ) = z̃_q^T Cov(β) z̃_q + σ²
          = σ² (z̃_q^T A^{-1} z̃_q + 1)

This decomposes uncertainty into two sources:
  • σ² z̃_q^T A^{-1} z̃_q  — uncertainty due to finite / noisy neighbour labels
  • σ²                    — irreducible observation noise

Training loss — Gaussian negative log-likelihood:
  L = ½ [(ŷ - y)² / var_y + log(var_y)]

which collapses to MSE when all observations are equally uncertain.
"""

import torch
import torch.nn as nn


class UncertainMatrixGWR(nn.Module):
    """
    WLS in K_z-projected TabPFN embedding space with Bayesian posterior
    uncertainty estimation.  No additional learnable parameters.

    Returns point prediction ŷ, predictive standard deviation σ, and
    local coefficients β for spatial analysis.
    """

    def __init__(self, lam: float = 1e-3, eps: float = 1e-6):
        super().__init__()
        self.lam = lam
        self.eps = eps   # numerical floor for variance

    def forward(self,
                z_query: torch.Tensor,
                K_z:     torch.Tensor,
                z_nbr:   torch.Tensor,
                y_nbr:   torch.Tensor,
                w:       torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z_query : (B, tabpfn_dim)
        K_z     : (B, tabpfn_dim, e)   e = z_proj_dim
        z_nbr   : (B, k, tabpfn_dim)
        y_nbr   : (B, k)
        w       : (B, k)               attention weights, sum ≈ 1

        Returns
        -------
        y_hat  : (B,)    — point predictions
        sigma  : (B,)    — predictive standard deviation
        beta   : (B, e)  — local MAP coefficients
        """
        B, k, d = z_nbr.shape
        e = K_z.shape[-1]

        # ── 1. Project z into regression subspace: Z̃ = z @ K_z ───────────
        z_q_tilde = torch.bmm(z_query.unsqueeze(1), K_z).squeeze(1)    # (B, e)
        Z_tilde   = torch.bmm(z_nbr, K_z)                               # (B, k, e)

        # ── 2. Build regularised normal matrix A = Z̃^T diag(w) Z̃ + λI ───
        w_sqrt = w.sqrt().unsqueeze(-1)          # (B, k, 1)
        Zw     = Z_tilde * w_sqrt                # (B, k, e)
        A      = torch.bmm(Zw.transpose(1, 2), Zw)  # (B, e, e)
        eye    = torch.eye(e, device=A.device, dtype=A.dtype).unsqueeze(0)
        A      = A + self.lam * eye              # (B, e, e)

        # ── 3. Right-hand side b = Z̃^T diag(w) y ─────────────────────────
        wy = (w * y_nbr).unsqueeze(-1)           # (B, k, 1)
        b  = torch.bmm(Z_tilde.transpose(1, 2), wy).squeeze(-1)  # (B, e)

        # ── 4. MAP estimate: β̂ = A^{-1} b ────────────────────────────────
        beta  = torch.linalg.solve(A, b)         # (B, e)

        # ── 5. Prediction ŷ = z̃_q · β̂ ────────────────────────────────────
        y_hat = (z_q_tilde * beta).sum(dim=-1)   # (B,)

        # ── 6. Local noise σ² from weighted residuals ──────────────────────
        # r_j = y_j - z̃_j · β̂   for each neighbour j
        # (B, k):  y_nbr - (Z̃ β̂ summed over e)
        y_pred_nbr = (Z_tilde * beta.unsqueeze(1)).sum(dim=-1)  # (B, k)
        residuals  = y_nbr - y_pred_nbr                          # (B, k)
        sigma2_obs = (w * residuals.pow(2)).sum(dim=-1)          # (B,)
        # clamp to avoid log(0) / division by zero
        sigma2_obs = sigma2_obs.clamp(min=self.eps)              # (B,)

        # ── 7. Posterior predictive variance ──────────────────────────────
        # Var(ŷ) = σ² (z̃_q^T A^{-1} z̃_q + 1)
        # Solve A v = z̃_q  (uses the same LU already computed by PyTorch)
        v         = torch.linalg.solve(A, z_q_tilde.unsqueeze(-1)).squeeze(-1)  # (B, e)
        quad_term = (z_q_tilde * v).sum(dim=-1)                  # (B,)   z̃_q^T A^{-1} z̃_q
        var_pred  = sigma2_obs * (quad_term + 1.0)               # (B,)
        sigma     = var_pred.clamp(min=self.eps).sqrt()          # (B,)

        return y_hat, sigma, beta


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian NLL loss for training GWF_U
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_nll_loss(y_hat: torch.Tensor,
                      sigma: torch.Tensor,
                      y_true: torch.Tensor,
                      beta_weight: float = 0.0) -> torch.Tensor:
    """
    Gaussian negative log-likelihood:
        L = ½ [(y_true - y_hat)² / σ² + log(σ²)]

    Parameters
    ----------
    y_hat       : (B,) — predictions
    sigma       : (B,) — predictive standard deviations
    y_true      : (B,) — ground-truth targets
    beta_weight : regularisation weight on σ² to prevent collapse to 0
                  (adds beta_weight * mean(σ²) to the loss)

    Returns
    -------
    scalar loss
    """
    var   = sigma.pow(2)
    nll   = 0.5 * ((y_true - y_hat).pow(2) / var + var.log())
    loss  = nll.mean()
    if beta_weight > 0.0:
        loss = loss + beta_weight * var.mean()
    return loss
