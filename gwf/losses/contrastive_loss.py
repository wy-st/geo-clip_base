"""
gwf/losses/contrastive_loss.py
================================
Geographic Contrastive Loss (L_contrast) — Phase 1 ONLY

Used during the bridge warm-up phase to pull nearby points together in the
fused embedding space. This aligns all 6 encoder channels into a spatially
coherent representation before joint training begins.

Loss (InfoNCE variant):
  For each edge (i, j) with weight w_ij (positive pair):
    pos_sim = w_ij * cosine_sim(h_i, h_j) / T
  Negatives: all other points k in the batch
    neg_sim = cosine_sim(h_i, h_k) / T
  L_contrast = mean( -pos_sim + log(Σ exp(neg_sim)) )

After Phase 1: this loss is discarded and never used again.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeographicContrastiveLoss(nn.Module):
    """
    InfoNCE-style geographic contrastive loss on h_fused embeddings.

    Args:
        temperature : softmax temperature (default 0.1)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.T = temperature

    def forward(
        self,
        h_fused:     torch.Tensor,   # (N, d)  fused embeddings
        edge_index:  torch.Tensor,   # (2, E)  spatial graph
        edge_weight: torch.Tensor,   # (E,)    learned weights ∈ [0,1]
    ) -> torch.Tensor:
        """Returns scalar contrastive loss."""
        N = h_fused.shape[0]
        dst = edge_index[0]
        src = edge_index[1]

        # Normalise embeddings
        h_norm = F.normalize(h_fused, dim=-1)  # (N, d)

        # Full similarity matrix (N, N) — used to get negatives
        sim_matrix = h_norm @ h_norm.t() / self.T  # (N, N)

        # Mask out the diagonal (self-similarity) for negatives
        mask_diag = torch.eye(N, dtype=torch.bool, device=h_fused.device)
        sim_matrix = sim_matrix.masked_fill(mask_diag, float("-inf"))

        # For each edge, compute positive similarity weighted by w_ij
        h_dst = h_norm[dst]              # (E, d)
        h_src = h_norm[src]              # (E, d)
        pos_cos = (h_dst * h_src).sum(-1) / self.T   # (E,)
        pos_sim = edge_weight * pos_cos               # (E,) weighted positive

        # Log-sum-exp over all negatives for destination nodes
        # log_denom[i] = log Σ_{k≠i} exp(sim(h_i, h_k) / T)
        log_denom = torch.logsumexp(sim_matrix[dst], dim=-1)  # (E,)

        # InfoNCE loss per edge
        loss_per_edge = -pos_sim + log_denom   # (E,)
        return loss_per_edge.mean()
