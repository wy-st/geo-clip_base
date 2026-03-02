"""
GWF — Geographical Weights Foundation Model (GNNWR-style, TabPFN embedding space)

Full pipeline for one forward pass (B query points, each with k neighbours):

  coords (B,2), x (B,p), nbr_x (B,k,p), nbr_y (B,k)
        │
        ├─ LocationFusion (GeoCLIP + SatCLIP, frozen)
        │       └─ e_loc  (B, loc_proj_dim)          ← 1 linear layer
        │
        ├─ TabPFNInContextEncoder (pretrained TabPFN, frozen)
        │       ├─ input : x_query (B,p) | x_nbr (B,k,p) | y_nbr (B,k)
        │       ├─ z_query (B, tabpfn_dim)            ← pre-MLP hidden state
        │       └─ z_nbr   (B, k, tabpfn_dim)         ← neighbours' pre-MLP states
        │
        ├─ Node projection: concat(z, e_loc) → h  (B, node_dim)  ← 1 linear
        │   (shared weights for query and neighbour nodes)
        │
        ├─ DynamicKernelGenerator  (GNNWR-style learned spatial weighting)
        │       ├─ K_z (B, tabpfn_dim, z_proj_dim)   — adaptive z-space projection
        │       │       generated via low-rank U @ V^T from h_query
        │       └─ w_i  (B, k)                        — spatial attention weights
        │
        └─ MatrixGWR  (WLS in the TabPFN embedding space)
                ├─ Z̃ = z_nbr @ K_z                   (project TabPFN embeddings)
                ├─ β_i = WLS(Z̃, y_nbr, w_i)          ← closed form, dim=z_proj_dim
                └─ ŷ_i = (z_query @ K_z) · β_i

Key design insight (from GNNWR):
  Spatial weights w_i are learned from contextual node representations h_i/h_j
  (encoding both spatial position and tabular context) rather than a fixed kernel.
  The WLS regression operates in the TabPFN hidden-feature space (z), which
  captures the in-context distribution of neighbours and is richer than raw x.

Trainable parameters live ONLY in:
  • LocationFusion.proj          (1 linear)
  • NodeProjection               (1 linear)
  • DynamicKernelGenerator       (3 linear — z_kernel, query, key heads)

Frozen:
  • GeoCLIP encoder              (pretrained)
  • SatCLIP proxy                (pretrained / RFF)
  • TabPFN PerFeatureTransformer (pretrained)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders       import LocationFusion
from .kernel         import DynamicKernelGenerator
from .regression     import MatrixGWR
from .tabpfn_encoder import TabPFNInContextEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Main GWF Model
# ─────────────────────────────────────────────────────────────────────────────

class GWF(nn.Module):
    """
    Geographical Weights Foundation Model.

    Parameters
    ----------
    feat_dim     : number of raw tabular features  (p) — used by TabPFN encoder
    loc_proj_dim : output dim of LocationFusion    (after GeoCLIP + SatCLIP fusion)
    node_dim     : node representation dimension   (input to DynamicKernelGenerator)
    z_proj_dim   : regression dimension in TabPFN embedding space
                   (z_proj_dim << tabpfn_dim; determines β dimension)
    kernel_rank  : low-rank factor r for K_z = U @ V^T
    attn_dim     : query/key dim for spatial attention weights
    wls_lambda   : ridge regularisation in WLS
    tabpfn_path  : path to a local TabPFN regressor .ckpt file;
                   None → try HuggingFace download
    """

    def __init__(
        self,
        feat_dim:      int        = 8,
        loc_proj_dim:  int        = 256,
        node_dim:      int        = 256,
        z_proj_dim:    int        = 64,
        kernel_rank:   int        = 4,
        attn_dim:      int        = 64,
        wls_lambda:    float      = 1e-3,
        tabpfn_path:   str | None = None,
        satclip_ckpt:  str | None = None,
    ):
        super().__init__()

        self.feat_dim   = feat_dim
        self.z_proj_dim = z_proj_dim

        # ── Location encoder (GeoCLIP + SatCLIP, frozen) ──────────────────
        self.loc_enc = LocationFusion(
            geo_dim=512, sat_dim=512, out_dim=loc_proj_dim,
            satclip_ckpt=satclip_ckpt)

        # ── TabPFN in-context encoder (pretrained, frozen) ─────────────────
        self.ctx_enc = TabPFNInContextEncoder(model_path=tabpfn_path)
        tabpfn_dim = self.ctx_enc.emb_dim   # TabPFN's ninp, e.g. 192

        # ── Node projection: [tabpfn_emb ‖ loc_proj] → node_dim ───────────
        # Shared weights: used for both query and neighbour nodes
        self.node_proj = nn.Linear(tabpfn_dim + loc_proj_dim, node_dim,
                                   bias=True)

        # ── Dynamic kernel matrix and attention weights (GNNWR-style) ──────
        self.kernel_gen = DynamicKernelGenerator(
            node_dim=node_dim,
            tabpfn_dim=tabpfn_dim,
            z_proj_dim=z_proj_dim,
            rank=kernel_rank,
            attn_dim=attn_dim)

        # ── Matrix WLS in TabPFN embedding space (no learnable parameters) ─
        self.gwr = MatrixGWR(lam=wls_lambda)

    # ─────────────────────────────────────────────────────────────────────────

    def _project_node(
        self,
        z:     torch.Tensor,   # (*, tabpfn_dim)   — TabPFN embedding
        e_loc: torch.Tensor,   # (*, loc_proj_dim) — location embedding
    ) -> torch.Tensor:
        """Shared node projection: concat(z, e_loc) → GELU → h (*, node_dim)."""
        return F.gelu(self.node_proj(torch.cat([z, e_loc], dim=-1)))

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
        y_hat : (B,)            — predictions
        beta  : (B, z_proj_dim) — local GWR coefficients in projected z-space
        """
        coord     = batch["coord"]       # (B, 2)
        x         = batch["x"]           # (B, p)
        nbr_coord = batch["nbr_coord"]   # (B, k, 2)
        nbr_x     = batch["nbr_x"]       # (B, k, p)
        nbr_y     = batch["nbr_y"]       # (B, k)
        nbr_dist  = batch["nbr_dist"]    # (B, k)

        B, k, p = nbr_x.shape

        # ── 1. TabPFN: context-aware embeddings (single forward call) ─────
        # Neighbours serve as TabPFN's in-context training set (with labels).
        # The query point is the "test" row — TabPFN never sees its label.
        #
        # z_query : (B, tabpfn_dim)    — query pre-MLP hidden state
        # z_nbr   : (B, k, tabpfn_dim) — neighbour pre-MLP hidden states
        z_query, z_nbr = self.ctx_enc(x, nbr_x, nbr_y)

        # ── 2. Location embeddings ─────────────────────────────────────────
        e_loc_query = self.loc_enc(coord)        # (B, loc_proj_dim)
        e_loc_nbr   = self.loc_enc(nbr_coord)    # (B, k, loc_proj_dim)

        # ── 3. Node projection (shared weights for query and neighbours) ───
        h_query = self._project_node(z_query, e_loc_query)     # (B, node_dim)

        # Flatten neighbours, project with shared weights, then reshape
        z_nbr_flat  = z_nbr.reshape(B * k, -1)               # (B*k, tabpfn_dim)
        e_loc_nbr_f = e_loc_nbr.reshape(B * k, -1)           # (B*k, loc_proj_dim)
        h_nbr = self._project_node(
            z_nbr_flat, e_loc_nbr_f
        ).reshape(B, k, -1)                                    # (B, k, node_dim)

        # ── 4. K_z: location-adaptive z-space projection, w: spatial weights
        # GNNWR-style: both are generated from the learned node representations
        K_z = self.kernel_gen.get_z_kernel_matrix(h_query)          # (B, tabpfn_dim, z_proj_dim)
        w   = self.kernel_gen.get_attention_weights(
            h_query, h_nbr, dist=nbr_dist)                          # (B, k)

        # ── 5. Matrix WLS in TabPFN embedding space ────────────────────────
        # Z̃ = z @ K_z  →  β = WLS(Z̃, y, w)  →  ŷ = (z_query @ K_z) · β
        y_hat, beta = self.gwr(z_query, K_z, z_nbr, nbr_y, w)      # (B,), (B,z_proj_dim)

        return y_hat, beta

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_with_betas(self, dataloader, device: str = "cpu"):
        """
        Run inference over a full dataset, collecting predictions and β vectors.

        Returns
        -------
        y_hat  : (N,)            — all predictions
        betas  : (N, z_proj_dim) — local GWR coefficients in projected z-space
        coords : (N, 2)          — query coordinates
        y_true : (N,)            — ground-truth targets
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
