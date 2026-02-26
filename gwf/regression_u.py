"""
GWF-U — Differentiable WLS with BNN-style Reparameterized Uncertainty

Two independent sources of uncertainty are modelled via the reparameterization
trick and estimated by Monte-Carlo sampling at inference time:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Source 1 — β coefficient uncertainty  (BNN-style reparameterization)
─────────────────────────────────────────────────────────────────────────────
The ridge-WLS solution has an exact Bayesian interpretation:

  Prior        : β ~ N(0, λ⁻¹ I)
  Likelihood   : y_j | z̃_j, β ~ N(z̃_j · β, σ²_obs)
  Normal matrix: A = Z̃ᵀ diag(w) Z̃ + λI
  Posterior    : β | y ~ N(μ_β, σ²_obs A⁻¹)
                 μ_β = A⁻¹ Z̃ᵀ diag(w) y   (MAP / posterior mean)

BNN reparameterization of the posterior:
  A = L Lᵀ              (Cholesky decomposition)
  β_sample = μ_β + σ_obs · L⁻ᵀ ε_β,    ε_β ~ N(0, I_e)

This is the Bayesian WLS posterior expressed as a reparameterized "BNN layer":
  • μ_β  plays the role of the mean weights
  • σ_obs · L⁻ᵀ  plays the role of the (data-adaptive) weight-noise matrix
  • ε_β  is the external noise variable (gradient flows through μ_β and σ_obs)

Unlike a standard BNN (where μ and σ are free parameters learned by backprop),
here both are analytically derived from the local data geometry — a principled,
data-informed Bayesian posterior that requires no additional parameters.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Source 2 — Spatial neighbourhood-range uncertainty  (in kernel.py)
─────────────────────────────────────────────────────────────────────────────
Handled upstream in DynamicKernelGenerator.get_attention_weights via a
learnable log_sigma_attn that perturbs the attention logits before softmax:
  scores_noisy = scores + exp(log_sigma_attn) · ε_attn,  ε_attn ~ N(0, I_k)
Different ε_attn realisations produce different spatial-weight vectors w,
propagating through WLS to different ŷ.  The variance of ŷ over ε_attn
samples (with ε_β fixed) is the spatial-range uncertainty.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training — pure MSE via reparameterized forward pass
─────────────────────────────────────────────────────────────────────────────
  ŷ_sample = z̃_q · β_sample
  L_train  = MSE(ŷ_sample, y_true)

No distributional assumption on y.  Gradients flow through μ_β (prediction
accuracy) and σ_obs (indirectly, by encouraging tighter local WLS fits).

Uncertainty decomposition at inference (law of total variance):
  σ²_total   = Var_{ε_attn, ε_β}(ŷ)
  σ²_β       = Var_{ε_β}(ŷ)          [ε_attn = 0, only β noise]
  σ²_spatial = σ²_total − σ²_β       [marginal spatial-range contribution]
"""

import torch
import torch.nn as nn


class UncertainMatrixGWR(nn.Module):
    """
    WLS in K_z-projected TabPFN embedding space with BNN-style reparameterized
    uncertainty for the local coefficient vector β.

    Forward behaviour
    -----------------
    • eps_beta = None  →  deterministic MAP prediction  (ŷ = z̃_q · μ_β)
    • eps_beta given   →  stochastic BNN sample         (ŷ = z̃_q · β_sample)

    Returns
    -------
    y_hat     : (B,)  — point prediction (MAP or sampled)
    sigma_obs : (B,)  — local WLS residual std  σ_obs = sqrt(Σ_j w_j r_j²)
                        used as the posterior noise scale; also informative
                        as a stand-alone aleatoric-noise estimate
    beta      : (B, e) — MAP coefficients μ_β  (always the posterior mean,
                         regardless of eps_beta, for spatial interpretability)
    """

    def __init__(self, lam: float = 1e-3, eps: float = 1e-6):
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(self,
                z_query:  torch.Tensor,
                K_z:      torch.Tensor,
                z_nbr:    torch.Tensor,
                y_nbr:    torch.Tensor,
                w:        torch.Tensor,
                eps_beta: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z_query  : (B, tabpfn_dim)
        K_z      : (B, tabpfn_dim, e)
        z_nbr    : (B, k, tabpfn_dim)
        y_nbr    : (B, k)
        w        : (B, k)              spatial attention weights (sum ≈ 1)
        eps_beta : (B, e) | None       external noise for β reparameterization;
                                       None → deterministic MAP forward pass
        """
        B, k, d = z_nbr.shape
        e = K_z.shape[-1]

        # ── 1. Project z into regression subspace: Z̃ = z @ K_z ───────────
        z_q_tilde = torch.bmm(z_query.unsqueeze(1), K_z).squeeze(1)   # (B, e)
        Z_tilde   = torch.bmm(z_nbr, K_z)                              # (B, k, e)

        # ── 2. Build regularised normal matrix A = Z̃ᵀ diag(w) Z̃ + λI ────
        w_sqrt = w.sqrt().unsqueeze(-1)          # (B, k, 1)
        Zw     = Z_tilde * w_sqrt                # (B, k, e)
        A      = torch.bmm(Zw.transpose(1, 2), Zw)  # (B, e, e)
        eye    = torch.eye(e, device=A.device, dtype=A.dtype).unsqueeze(0)
        A      = A + self.lam * eye              # (B, e, e)

        # ── 3. Right-hand side b = Z̃ᵀ diag(w) y ──────────────────────────
        wy = (w * y_nbr).unsqueeze(-1)           # (B, k, 1)
        b  = torch.bmm(Z_tilde.transpose(1, 2), wy).squeeze(-1)  # (B, e)

        # ── 4. MAP estimate: μ_β = A⁻¹ b ──────────────────────────────────
        mu_beta = torch.linalg.solve(A, b)       # (B, e)

        # ── 5. Local noise σ_obs from weighted residuals ───────────────────
        y_pred_nbr = (Z_tilde * mu_beta.unsqueeze(1)).sum(dim=-1)  # (B, k)
        residuals  = y_nbr - y_pred_nbr                             # (B, k)
        sigma2_obs = (w * residuals.pow(2)).sum(dim=-1)             # (B,)
        sigma2_obs = sigma2_obs.clamp(min=self.eps)
        sigma_obs  = sigma2_obs.sqrt()                              # (B,)

        # ── 6. BNN reparameterization of β (optional) ─────────────────────
        # Posterior: β | y ~ N(μ_β, σ²_obs A⁻¹)
        # Reparameterize: β = μ_β + σ_obs · L⁻ᵀ ε_β
        #   where A = L Lᵀ  →  chol(A⁻¹) = L⁻ᵀ  (upper-triangular)
        if eps_beta is not None:
            L = torch.linalg.cholesky(A)         # (B, e, e), lower triangular
            # Solve Lᵀ v = ε_β  →  v = L⁻ᵀ ε_β
            v = torch.linalg.solve_triangular(
                L.mT,                            # upper triangular
                eps_beta.unsqueeze(-1),          # (B, e, 1)
                upper=True,
            ).squeeze(-1)                        # (B, e)
            beta = mu_beta + sigma_obs.unsqueeze(-1) * v
        else:
            beta = mu_beta

        # ── 7. Prediction ŷ = z̃_q · β ─────────────────────────────────────
        y_hat = (z_q_tilde * beta).sum(dim=-1)   # (B,)

        return y_hat, sigma_obs, mu_beta


# ─────────────────────────────────────────────────────────────────────────────
# Legacy analytical NLL loss (kept for research comparisons)
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_nll_loss(y_hat: torch.Tensor,
                      sigma: torch.Tensor,
                      y_true: torch.Tensor,
                      beta_weight: float = 0.0) -> torch.Tensor:
    """
    Gaussian negative log-likelihood:
        L = ½ [(y_true - y_hat)² / σ² + log(σ²)]

    Kept for research comparisons with the BNN-MSE approach.
    The default GWF-U training now uses pure MSE via reparameterized forward.
    """
    var  = sigma.pow(2)
    nll  = 0.5 * ((y_true - y_hat).pow(2) / var + var.log())
    loss = nll.mean()
    if beta_weight > 0.0:
        loss = loss + beta_weight * var.mean()
    return loss
