"""
GWF — Spatial Context Module

Replaces the WLS + DynamicKernelGenerator pair with a single module that:

  1. Injects y_nbr into neighbour node representations → h_aug
     (allows spatial attention to consider the local label distribution)

  2. Computes cross-attention weights from h_query → h_aug
     with an optional geographic distance prior

  3. Aggregates neighbours via weighted sum → context vector

  4. Generates local β coefficients:
        β_i = beta_head(h_query + context)   ∈ R^{z_proj_dim}
     β is a spatially-varying vector; can be visualised on the map.

  5. Predicts via a shared static projection W ∈ R^{tabpfn_dim × z_proj_dim}:
        ŷ_i = (z_query @ W) · β_i

This design decouples k completely from model parameters:
  • k is a pure hyperparameter (no z_proj_dim ≤ k/2 constraint)
  • The β generation is stable for any k ≥ 1

Trainable parameters:
  y_proj   : 1 → y_inject_dim  (tiny — embeds scalar y into feature space)
  y_inject : y_inject_dim → H  (add to h_nbr)
  query/key heads               (spatial attention)
  W_static                      (tabpfn_dim → z_proj_dim, shared projection)
  beta_head : H → z_proj_dim   (generates local β from fused representation)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GWRContextModule(nn.Module):
    """
    Context-driven β generation for Geographically Weighted Regression.

    Parameters
    ----------
    node_dim      : dimension of the node representation h_i (from node_proj)
    tabpfn_dim    : TabPFN hidden-state dimension (D)
    z_proj_dim    : β and projection dimension (E)
                    No constraint on k — freely set as a model hyperparameter.
    attn_dim      : query/key dimension for spatial attention (A)
    y_inject_dim  : small embedding dimension for y_nbr injection (default 8)
    """

    def __init__(self,
                 node_dim:     int,
                 tabpfn_dim:   int,
                 z_proj_dim:   int = 32,
                 attn_dim:     int = 64,
                 y_inject_dim: int = 8):
        super().__init__()
        self.tabpfn_dim   = tabpfn_dim
        self.z_proj_dim   = z_proj_dim
        self.attn_dim     = attn_dim
        self.y_inject_dim = y_inject_dim

        # ── y-injection into neighbour representations ────────────────────
        # Projects scalar y_nbr → y_inject_dim, then adds to h_nbr
        self.y_proj   = nn.Sequential(nn.Linear(1, y_inject_dim), nn.Tanh())
        self.y_inject = nn.Linear(y_inject_dim, node_dim, bias=False)

        # ── Spatial attention (query attends to y-augmented neighbours) ───
        self.query_head = nn.Linear(node_dim, attn_dim, bias=False)
        self.key_head   = nn.Linear(node_dim, attn_dim, bias=False)
        nn.init.xavier_uniform_(self.query_head.weight)
        nn.init.xavier_uniform_(self.key_head.weight)

        # ── Shared static projection: tabpfn_dim → z_proj_dim ─────────────
        self.W_static = nn.Parameter(torch.zeros(tabpfn_dim, z_proj_dim))
        nn.init.normal_(self.W_static, std=0.02)

        # ── β generation head: fused representation → local coefficients ──
        self.beta_head = nn.Linear(node_dim, z_proj_dim, bias=True)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self,
                h_query: torch.Tensor,         # (B, H)   query node repr
                h_nbr:   torch.Tensor,         # (B, k, H) neighbour node repr
                y_nbr:   torch.Tensor,         # (B, k)   neighbour labels
                z_query: torch.Tensor,         # (B, D)   query TabPFN emb
                dist:    torch.Tensor | None,  # (B, k) | None  geographic dist
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        y_hat  : (B,)            — point prediction
        beta   : (B, z_proj_dim) — local GWR coefficients (spatially varying)
        w      : (B, k)          — spatial attention weights (for inspection)
        """
        # 1. y-inject: augment h_nbr with local label information
        y_feat = self.y_proj(y_nbr.unsqueeze(-1))     # (B, k, y_inject_dim)
        h_aug  = h_nbr + self.y_inject(y_feat)         # (B, k, H)

        # 2. Attention scores: h_query ↔ y-augmented neighbours
        q = self.query_head(h_query)                   # (B, A)
        k = self.key_head(h_aug)                       # (B, k, A)
        scale  = math.sqrt(self.attn_dim)
        scores = torch.einsum("ba,bka->bk", q, k) / scale   # (B, k)

        if dist is not None:
            d_norm  = dist / (dist.max(dim=-1, keepdim=True).values + 1e-8)
            scores  = scores - d_norm

        w = F.softmax(scores, dim=-1)                  # (B, k)

        # 3. Weighted aggregation → context vector
        context = torch.einsum("bk,bkh->bh", w, h_aug)  # (B, H)

        # 4. β generation from query + context
        h_fused = h_query + context                     # (B, H)  residual
        beta    = self.beta_head(h_fused)               # (B, E)

        # 5. Predict via dot product in projected z-space
        z_proj  = z_query @ self.W_static               # (B, E)
        y_hat   = (z_proj * beta).sum(dim=-1)           # (B,)

        return y_hat, beta, w
