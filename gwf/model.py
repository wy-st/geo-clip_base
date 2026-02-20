"""
GWF — Geographical Weights Foundation Model

Full pipeline for one forward pass (B query points, each with k neighbours):

  coords (B,2), x (B,p)
        │
        ├─ LocationFusion (GeoCLIP + SatCLIP, frozen)
        │       └─ e_loc  (B, loc_proj_dim)          ← 1 linear layer
        │
        ├─ TabPFN-inspired In-Context Encoder
        │       └─ z  (B, feat_emb_dim)              ← 1 linear + 1 cross-attn
        │
        ├─ Node projection: concat(z, e_loc) → h  (B, node_dim)  ← 1 linear
        │
        ├─ DynamicKernelGenerator
        │       ├─ K_i (B, p, p)   — feature transformation matrix ← 1 linear
        │       └─ w_i (B, k)      — spatial attention weights     ← 1 linear q/k
        │
        └─ MatrixGWR
                ├─ X̃ = X_nbr @ K_i
                ├─ β_i = WLS(X̃, y_nbr, w_i)          ← closed form
                └─ ŷ_i = (x_i @ K_i) · β_i

Trainable parameters live ONLY in:
  • LocationFusion.proj          (1 linear)
  • InContextEncoder.feat_proj   (1 linear)
  • InContextEncoder.cross_attn  (MultiheadAttention — lightweight)
  • NodeProjection               (1 linear)
  • DynamicKernelGenerator       (3 linear — kernel, query, key heads)

The frozen base models (GeoCLIP, SatCLIP proxy) are never touched.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders   import LocationFusion
from .kernel     import DynamicKernelGenerator
from .regression import MatrixGWR


# ──────────────────────────────────────────────────────────────────────────────
# TabPFN-inspired In-Context Feature Encoder
# ──────────────────────────────────────────────────────────────────────────────

class InContextEncoder(nn.Module):
    """
    Encodes tabular features with neighbourhood context (TabPFN-inspired).

    For each query point i:
      1. Linear projection: x_i → z_i                    (raw features → embedding)
      2. Cross-attention over k neighbours:
             z_i attends to {z_j | j ∈ N(i)}
         so z_i becomes context-aware without any additional MLP layers.

    This mimics TabPFN's in-context learning: the query's representation
    is shaped by the distribution of local training samples.

    Input
    -----
    x_query : (B, p)
    x_nbr   : (B, k, p)

    Output
    ------
    z_query : (B, feat_emb_dim)   context-aware feature embedding
    z_nbr   : (B, k, feat_emb_dim) neighbour embeddings (needed for node repr.)
    """

    def __init__(self, feat_dim: int, emb_dim: int, n_heads: int = 4):
        super().__init__()
        # Single linear projection (no bias to keep it minimal)
        self.feat_proj = nn.Linear(feat_dim, emb_dim, bias=False)

        # PyTorch MultiheadAttention is a single module, not an MLP
        # batch_first=True: (B, seq, dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=n_heads,
            dropout=0.0, batch_first=True)

    def forward(self, x_query: torch.Tensor,
                x_nbr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, k, p = x_nbr.shape

        # Project all points to embedding space
        z_q   = self.feat_proj(x_query)        # (B, emb_dim)
        z_nbr = self.feat_proj(x_nbr.reshape(B * k, p)).reshape(B, k, -1)  # (B, k, emb_dim)

        # Cross-attention: query=z_q, key/value=z_nbr
        z_q_seq = z_q.unsqueeze(1)              # (B, 1, emb_dim) — sequence of length 1
        ctx, _  = self.cross_attn(z_q_seq, z_nbr, z_nbr)  # (B, 1, emb_dim)
        z_out   = z_q + ctx.squeeze(1)         # residual connection

        return z_out, z_nbr


# ──────────────────────────────────────────────────────────────────────────────
# Main GWF Model
# ──────────────────────────────────────────────────────────────────────────────

class GWF(nn.Module):
    """
    Geographical Weights Foundation Model.

    Parameters correspond to GWFConfig fields.
    """

    def __init__(self,
                 feat_dim:     int   = 8,
                 feat_emb_dim: int   = 64,
                 loc_proj_dim: int   = 256,
                 node_dim:     int   = 256,
                 kernel_rank:  int   = 4,
                 attn_dim:     int   = 64,
                 wls_lambda:   float = 1e-3):
        super().__init__()

        self.feat_dim = feat_dim

        # ── Location encoder (GeoCLIP + SatCLIP, frozen) ──────────────────
        self.loc_enc = LocationFusion(
            geo_dim=512, sat_dim=512, out_dim=loc_proj_dim)

        # ── In-context tabular encoder (TabPFN-inspired) ───────────────────
        self.ctx_enc = InContextEncoder(feat_dim, feat_emb_dim, n_heads=4)

        # ── Node projection: [feat_emb ‖ loc_proj] → node_dim (1 linear) ──
        self.node_proj = nn.Linear(feat_emb_dim + loc_proj_dim, node_dim,
                                   bias=True)

        # ── Dynamic kernel & attention weights ────────────────────────────
        self.kernel_gen = DynamicKernelGenerator(
            node_dim=node_dim,
            feat_dim=feat_dim,
            rank=kernel_rank,
            attn_dim=attn_dim)

        # ── Matrix WLS ────────────────────────────────────────────────────
        self.gwr = MatrixGWR(lam=wls_lambda)

    # ─────────────────────────────────────────────────────────────────────────

    def _encode_node(self,
                     coords:   torch.Tensor,   # (*, 2)
                     x:        torch.Tensor,   # (*, p)
                     x_nbr:    torch.Tensor,   # (B, k, p)   only for query batch
                     is_query: bool = True
                     ) -> torch.Tensor:
        """Shared node encoding used for both query and neighbour nodes."""
        e_loc = self.loc_enc(coords)           # (*, loc_proj_dim)

        if is_query:
            z, _ = self.ctx_enc(x, x_nbr)     # (B, feat_emb_dim)
        else:
            # For neighbours: no cross-attn (they are the context themselves)
            z = self.ctx_enc.feat_proj(x)      # (*, feat_emb_dim)

        node = torch.cat([z, e_loc], dim=-1)   # (*, feat_emb + loc_proj)
        return F.gelu(self.node_proj(node))    # (*, node_dim)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch : dict with keys
            coord     : (B, 2)
            x         : (B, p)
            nbr_coord : (B, k, 2)
            nbr_x     : (B, k, p)
            nbr_y     : (B, k)
            nbr_dist  : (B, k)

        Returns
        -------
        y_hat : (B,)   — predictions
        beta  : (B, p) — local GWR coefficients (for analysis)
        """
        coord     = batch["coord"]       # (B, 2)
        x         = batch["x"]           # (B, p)
        nbr_coord = batch["nbr_coord"]   # (B, k, 2)
        nbr_x     = batch["nbr_x"]       # (B, k, p)
        nbr_y     = batch["nbr_y"]       # (B, k)
        nbr_dist  = batch["nbr_dist"]    # (B, k)

        B, k, p = nbr_x.shape

        # ── 1. Encode query nodes ─────────────────────────────────────────
        h_query = self._encode_node(coord, x, nbr_x, is_query=True)  # (B, node_dim)

        # ── 2. Encode neighbour nodes (shared weights, no cross-attn) ─────
        nbr_x_flat   = nbr_x.reshape(B * k, p)
        nbr_co_flat  = nbr_coord.reshape(B * k, 2)
        h_nbr_flat   = self._encode_node(nbr_co_flat, nbr_x_flat,
                                         None, is_query=False)         # (B*k, node_dim)
        h_nbr        = h_nbr_flat.reshape(B, k, -1)                   # (B, k, node_dim)

        # ── 3. Dynamic kernel matrix K_i and attention weights w_i ────────
        K = self.kernel_gen.get_kernel_matrix(h_query)               # (B, p, p)
        w = self.kernel_gen.get_attention_weights(h_query, h_nbr,
                                                   dist=nbr_dist)    # (B, k)

        # ── 4. Matrix WLS → prediction + local coefficients ───────────────
        y_hat, beta = self.gwr(x, K, nbr_x, nbr_y, w)               # (B,), (B,p)

        return y_hat, beta

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_with_betas(self, dataloader, device: str = "cpu"):
        """
        Run inference over a full dataset, collecting predictions and β vectors.

        Returns
        -------
        y_hat  : (N,)    — all predictions
        betas  : (N, p)  — local GWR coefficients
        coords : (N, 2)  — query coordinates
        y_true : (N,)    — ground-truth targets
        """
        self.eval()
        all_yhat, all_beta, all_coord, all_y = [], [], [], []
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            yh, beta = self(batch)
            all_yhat.append(yh.cpu())
            all_beta.append(beta.cpu())
            all_coord.append(batch["coord"].cpu())
            all_y.append(batch["y"].cpu())

        return (torch.cat(all_yhat),
                torch.cat(all_beta),
                torch.cat(all_coord),
                torch.cat(all_y))
