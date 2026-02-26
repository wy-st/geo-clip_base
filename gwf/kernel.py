"""
GWF — Dynamic Kernel Generator (GNNWR-style)

Inspired by GNNWR (Du et al., IJGIS 2020): a neural network learns the
spatial weight function from contextual node representations, replacing the
fixed kernel of classical GWR.

Given the aggregated node representation h_i (TabPFN embedding + location),
this module produces:

  1. K_z ∈ R^{tabpfn_dim × z_proj_dim}  — low-rank adaptive projection
       K_z = U_i @ V_i^T,   U_i ∈ R^{tabpfn_dim × rank},
                              V_i ∈ R^{z_proj_dim × rank}
     Applied as:  Z̃ = z_nbr @ K_z   (project TabPFN hidden features)
     This projects context-aware TabPFN embeddings into a location-adaptive
     subspace for the WLS regression.

  2. w_ij ∈ R^{k}  — scalar spatial attention weights over neighbours,
     computed via cross-attention:  q_i (from h_i) · k_j (from h_j)
     Softmax-normalised so Σ_j w_ij = 1.

Both outputs are produced by a single linear layer each, preserving the
rich representations from the frozen base models (TabPFN + GeoCLIP).

Spatial neighbourhood-range uncertainty
─────────────────────────────────────────
get_attention_weights accepts an optional eps: (B, k) noise tensor.
When provided, the attention logits are perturbed before softmax:

  scores_noisy = scores + exp(log_sigma_attn) · ε,   ε ~ N(0, I_k)

log_sigma_attn is a single learnable scalar (global across queries).
Different ε realisations produce different w → different WLS solutions →
different ŷ.  The variance of ŷ over ε samples quantifies how sensitive
the prediction is to uncertainty in *which neighbours are relevant* — i.e.
the spatial neighbourhood-range uncertainty requested by the boss.

Intuition: in a dense, spatially homogeneous region, all neighbours give
similar predictions regardless of w, so σ_spatial ≈ 0.  In a transition
zone (e.g. urban/rural boundary), the choice of neighbourhood boundary
matters greatly, so σ_spatial is large.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicKernelGenerator(nn.Module):
    """
    Parameters
    ----------
    node_dim    : dimension of the aggregated node representation h_i
    tabpfn_dim  : TabPFN hidden-state dimension (emb_dim from TabPFNInContextEncoder)
    z_proj_dim  : projection dimension for WLS regression (z_proj_dim << tabpfn_dim)
    rank        : low-rank factorisation rank r for K_z = U @ V^T
    attn_dim    : query/key dimension for spatial attention weights
    """

    def __init__(self, node_dim: int, tabpfn_dim: int,
                 z_proj_dim: int = 64, rank: int = 4, attn_dim: int = 64):
        super().__init__()
        self.tabpfn_dim = tabpfn_dim
        self.z_proj_dim = z_proj_dim
        self.rank       = rank
        self.attn_dim   = attn_dim

        # ── K_z: (tabpfn_dim, z_proj_dim) low-rank factorisation ──────────
        # h_i → (U_i, V_i)  where K_z = U_i @ V_i^T
        # U_i ∈ R^{tabpfn_dim × rank},  V_i ∈ R^{z_proj_dim × rank}
        # Output size: (tabpfn_dim + z_proj_dim) * rank
        self.z_kernel_head = nn.Linear(
            node_dim, (tabpfn_dim + z_proj_dim) * rank, bias=False)

        # ── Spatial attention: query from h_i, key from neighbour h_j ─────
        self.query_head = nn.Linear(node_dim, attn_dim, bias=False)
        self.key_head   = nn.Linear(node_dim, attn_dim, bias=False)

        # ── Spatial range uncertainty: learnable log-scale for logit noise ─
        # Initialised to -1 (σ_attn ≈ 0.37) — small enough not to dominate
        # the learned attention scores at the start of training.
        self.log_sigma_attn = nn.Parameter(torch.tensor(-1.0))

        self._init_weights()

    def _init_weights(self):
        # Small init so K_z starts near zero — the regression begins close to
        # raw z features before learning location-specific projections.
        nn.init.normal_(self.z_kernel_head.weight, std=0.01)
        nn.init.xavier_uniform_(self.query_head.weight)
        nn.init.xavier_uniform_(self.key_head.weight)

    # ── K_z generation ───────────────────────────────────────────────────────

    def get_z_kernel_matrix(self, h: torch.Tensor) -> torch.Tensor:
        """
        Generate the location-adaptive z-space projection matrix.

        h   : (B, node_dim)
        Returns K_z : (B, tabpfn_dim, z_proj_dim)   low-rank  K_z = U @ V^T

        The projection maps TabPFN hidden features from tabpfn_dim → z_proj_dim
        in a location-aware manner, allowing each query point to select the
        most relevant dimensions of the in-context embeddings for local regression.
        """
        d, e, r = self.tabpfn_dim, self.z_proj_dim, self.rank
        uv = self.z_kernel_head(h)                    # (B, (d+e)*r)
        U  = uv[:, :d * r].reshape(-1, d, r)          # (B, tabpfn_dim, rank)
        V  = uv[:, d * r:].reshape(-1, e, r)          # (B, z_proj_dim, rank)
        K_z = torch.matmul(U, V.transpose(-1, -2))    # (B, tabpfn_dim, z_proj_dim)
        return K_z

    # ── Attention weights w_ij ───────────────────────────────────────────────

    def get_attention_weights(self,
                              h_query: torch.Tensor,
                              h_keys:  torch.Tensor,
                              dist:    torch.Tensor | None = None,
                              eps:     torch.Tensor | None = None,
                              ) -> torch.Tensor:
        """
        Compute learned spatial attention weights over neighbours (GNNWR-style).

        Unlike GNNWR's SWNN (which takes a distance vector as input), we use
        cross-attention between richer node representations that encode both
        spatial position (via GeoCLIP/SatCLIP) and tabular context (via TabPFN).

        h_query : (B, node_dim)
        h_keys  : (B, k, node_dim)
        dist    : (B, k) optional geographic distances — added as a soft prior
                  so nearer neighbours still tend to get higher weight
                  (distance-decay inductive bias, learnable to override)
        eps     : (B, k) | None — external noise for spatial-range
                  reparameterization.  When provided, logits are perturbed:
                    scores_noisy = scores + exp(log_sigma_attn) · eps
                  Different eps realisations yield different w → different
                  WLS solutions, enabling MC estimation of spatial-range
                  uncertainty in predict_with_uncertainty().

        Returns w : (B, k)  — spatial weights, sum to 1
        """
        q = self.query_head(h_query)                     # (B, attn_dim)
        k = self.key_head(h_keys)                        # (B, k, attn_dim)

        # Scaled dot-product attention
        scale  = math.sqrt(self.attn_dim)
        scores = torch.einsum("bd,bkd->bk", q, k) / scale  # (B, k)

        # Optional distance decay prior
        if dist is not None:
            # Normalise distances to [0,1] and subtract (closer → less penalty)
            d_norm = dist / (dist.max(dim=-1, keepdim=True).values + 1e-8)
            scores = scores - d_norm                        # soft distance bias

        # Spatial-range reparameterization: perturb logits before softmax
        if eps is not None:
            scores = scores + self.log_sigma_attn.exp() * eps

        return F.softmax(scores, dim=-1)                 # (B, k)
