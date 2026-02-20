"""
GWF — Differentiable Matrix-Form WLS (Geographically Weighted Regression)

Standard GWR solves, for each location i:

    β_i  =  (X_N^T W_i X_N)^{-1}  X_N^T W_i y_N       (scalar W_i)

We extend this by replacing raw X_N with K_i-transformed features:

    X̃_N  =  X_N @ K_i                                   (feature mixing)
    β_i  =  (X̃_N^T diag(w_i) X̃_N + λI)^{-1}  X̃_N^T diag(w_i) y_N

where:
  • K_i ∈ R^{p×p}   — dynamic feature transformation (low-rank, location-aware)
  • w_i ∈ R^{k}     — scalar attention weights over k neighbours (learned)
  • λ               — ridge regularisation

This gives β_i ∈ R^p — a high-dimensional local coefficient vector that
can be t-SNE'd to 1-D/2-D for spatial visualisation on the map.

Prediction at query point i:
    ŷ_i  =  (x_i @ K_i) · β_i
"""

import torch
import torch.nn as nn


class MatrixGWR(nn.Module):
    """
    Closed-form, differentiable WLS in the kernel-transformed feature space.
    No trainable parameters here — all parameters live in DynamicKernelGenerator.
    """

    def __init__(self, lam: float = 1e-3):
        super().__init__()
        self.lam = lam

    def forward(self,
                x_query: torch.Tensor,
                K:       torch.Tensor,
                X_nbr:  torch.Tensor,
                y_nbr:  torch.Tensor,
                w:      torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_query : (B, p)          — tabular features of query points
        K       : (B, p, p)       — dynamic feature transformation matrices
        X_nbr   : (B, k, p)       — neighbour tabular features
        y_nbr   : (B, k)          — neighbour targets
        w       : (B, k)          — attention weights (sum ≈ 1)

        Returns
        -------
        y_hat : (B,)              — predictions
        beta  : (B, p)            — local coefficients (for analysis / t-SNE)
        """
        B, k, p = X_nbr.shape

        # ── 1. Transform features: X̃ = X @ K ─────────────────────────────
        # x_query:  (B, p) → (B, 1, p) @ (B, p, p) → (B, 1, p) → (B, p)
        x_q_tilde = torch.bmm(x_query.unsqueeze(1), K).squeeze(1)       # (B, p)
        # X_nbr:   (B, k, p) @ (B, p, p) → (B, k, p)
        X_tilde = torch.bmm(X_nbr, K)                                    # (B, k, p)

        # ── 2. Weighted normal equations ───────────────────────────────────
        # W = diag(w)  so  X̃^T W X̃ = Σ_j w_j · x̃_j x̃_j^T
        # Efficient: use broadcasting
        w_sqrt = w.sqrt().unsqueeze(-1)          # (B, k, 1)
        Xw = X_tilde * w_sqrt                    # (B, k, p)  ← X̃ ⊙ √w (per row)

        # X̃^T W X̃ = Xw^T Xw
        A = torch.bmm(Xw.transpose(1, 2), Xw)   # (B, p, p)

        # Ridge: A + λI
        eye = torch.eye(p, device=A.device, dtype=A.dtype).unsqueeze(0)
        A = A + self.lam * eye                   # (B, p, p)

        # X̃^T W y = Σ_j w_j · x̃_j · y_j
        wy = (w * y_nbr).unsqueeze(-1)           # (B, k, 1)
        b  = torch.bmm(X_tilde.transpose(1, 2), wy).squeeze(-1)  # (B, p)

        # ── 3. Solve: β_i = A^{-1} b ──────────────────────────────────────
        # torch.linalg.solve is differentiable and numerically stable
        beta = torch.linalg.solve(A, b)          # (B, p)

        # ── 4. Predict: ŷ_i = (x_i @ K_i) · β_i ─────────────────────────
        y_hat = (x_q_tilde * beta).sum(dim=-1)   # (B,)

        return y_hat, beta
