"""
gwf/losses/smooth_loss.py
==========================
Spatial Smoothness Loss (L_smooth)

Enforces Tobler's First Law: nearby points should have similar regression
coefficients. Penalises Beta coefficient differences along graph edges,
weighted by learned edge weights w_ij.

L_smooth = (1/|E|) Σ_{(i,j)∈E} w_ij · [ ||sigma_i - sigma_j||²
                                          + 0.1 · ||U_i - U_j||²_F ]

- sigma captures the magnitude of each low-rank component → weighted 1.0
- U captures the direction of each component → weighted 0.1
- V is not compared (saves computation; sigma+U capture most variation)
"""

import torch
import torch.nn as nn


class SpatialSmoothnessLoss(nn.Module):
    """
    Penalises large differences in Beta coefficients between adjacent nodes.

    Args:
        sigma_weight : weight for sigma difference  (default 1.0)
        u_weight     : weight for U difference      (default 0.1)
    """

    def __init__(self, sigma_weight: float = 1.0, u_weight: float = 0.1):
        super().__init__()
        self.sigma_weight = sigma_weight
        self.u_weight     = u_weight

    def forward(
        self,
        beta:        dict[str, torch.Tensor],  # {"U":(N,d,r), "sigma":(N,r), "V":(N,r,d)}
        edge_index:  torch.Tensor,             # (2, E)
        edge_weight: torch.Tensor,             # (E,)
    ) -> torch.Tensor:
        """Returns a scalar smooth loss."""
        U     = beta["U"]      # (N, d, r)
        sigma = beta["sigma"]  # (N, r)

        dst = edge_index[0]   # target nodes
        src = edge_index[1]   # source nodes

        # Sigma difference: ||sigma_i - sigma_j||²  per edge → (E,)
        sigma_diff = (sigma[dst] - sigma[src]).pow(2).sum(-1)   # (E,)

        # U difference (Frobenius): ||U_i - U_j||²_F per edge → (E,)
        U_diff = (U[dst] - U[src]).pow(2).sum([-2, -1])         # (E,)

        # Weighted combination per edge
        per_edge = edge_weight * (
            self.sigma_weight * sigma_diff
            + self.u_weight   * U_diff
        )

        return per_edge.mean()
