"""
GWF — Dynamic Kernel Matrix Generator

Given the aggregated node representation h_i (from the graph propagation),
this module produces:

  1. K_i ∈ R^{p×p}  — low-rank feature transformation matrix
       K_i = U_i @ V_i^T,   U_i, V_i ∈ R^{p × rank}
     Applied as:  X̃ = X @ K_i   (matrix multiply, not scalar multiply)
     This lets the kernel mix features in a location-aware way.

  2. w_ij ∈ R^{k}  — scalar spatial attention weights over neighbours,
     computed via cross-attention:  q_i (query from h_i) · k_j (key from h_j)
     Result is softmax-normalised so Σ_j w_ij = 1.

Both outputs are produced by a SINGLE linear layer each (no hidden layers),
preserving the rich representation learned by the frozen base models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicKernelGenerator(nn.Module):
    """
    Parameters
    ----------
    node_dim  : dimension of the aggregated node representation h_i
    feat_dim  : number of tabular features p (size of K_i matrix)
    rank      : low-rank factorisation rank r  (r << p)
    attn_dim  : query/key dimension for spatial attention weights
    """

    def __init__(self, node_dim: int, feat_dim: int,
                 rank: int = 4, attn_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.rank     = rank
        self.attn_dim = attn_dim

        # ── Kernel matrix K_i = U_i @ V_i^T ──────────────────────────────
        # Output size: 2 * p * r  (U and V concatenated)
        self.kernel_head = nn.Linear(node_dim, 2 * feat_dim * rank, bias=False)

        # ── Spatial attention: query from h_i, key from neighbour h_j ─────
        self.query_head = nn.Linear(node_dim, attn_dim, bias=False)
        self.key_head   = nn.Linear(node_dim, attn_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Small init so K_i starts close to identity-like
        nn.init.normal_(self.kernel_head.weight, std=0.01)
        nn.init.xavier_uniform_(self.query_head.weight)
        nn.init.xavier_uniform_(self.key_head.weight)

    # ── K_i generation ───────────────────────────────────────────────────────

    def get_kernel_matrix(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : (..., node_dim)
        Returns K : (..., p, p)  low-rank matrix  K = U @ V^T

        The identity residual  K_i = I + U_i @ V_i^T  is used so the
        transformation starts as the identity and only learns deviations.
        This is crucial for preserving the base-model representations.
        """
        p, r = self.feat_dim, self.rank
        uv = self.kernel_head(h)                        # (..., 2*p*r)
        U, V = uv[..., :p*r], uv[..., p*r:]
        U = U.reshape(*h.shape[:-1], p, r)              # (..., p, r)
        V = V.reshape(*h.shape[:-1], p, r)              # (..., p, r)
        K = torch.matmul(U, V.transpose(-1, -2))        # (..., p, p)

        # Identity residual: preserve original features by default
        eye = torch.eye(p, device=h.device, dtype=h.dtype)
        for _ in range(K.dim() - 2):
            eye = eye.unsqueeze(0)
        return eye + K                                   # (..., p, p)

    # ── Attention weights w_ij ───────────────────────────────────────────────

    def get_attention_weights(self,
                              h_query: torch.Tensor,
                              h_keys:  torch.Tensor,
                              dist:    torch.Tensor | None = None
                              ) -> torch.Tensor:
        """
        Compute softmax attention weights over neighbours.

        h_query : (B, node_dim)
        h_keys  : (B, k, node_dim)
        dist    : (B, k) optional geographic distances — added as a bias
                  so nearer neighbours still tend to get higher weight
                  (distance-decay inductive bias, learnable to override)

        Returns w : (B, k)  — attention weights, sum to 1
        """
        q = self.query_head(h_query)                     # (B, attn_dim)
        k = self.key_head(h_keys)                        # (B, k, attn_dim)

        # Scaled dot-product attention
        scale = math.sqrt(self.attn_dim)
        scores = torch.einsum("bd,bkd->bk", q, k) / scale  # (B, k)

        # Optional distance decay prior
        if dist is not None:
            # Normalise distances to [0,1] and subtract (closer → less penalty)
            d_norm = dist / (dist.max(dim=-1, keepdim=True).values + 1e-8)
            scores = scores - d_norm                        # soft distance bias

        return F.softmax(scores, dim=-1)                 # (B, k)
