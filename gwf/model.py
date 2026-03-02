"""
GWF — Geographical Weights Foundation Model  (v2)

Pipeline (B query points, each with k neighbours):

  coords (B,2), x (B,p), nbr_x (B,k,p), nbr_y (B,k)
        │
        ├─ LocationFusion (GeoCLIP, frozen)
        │       └─ e_loc  (B, loc_proj_dim)
        │
        ├─ TabPFNInContextEncoder (pretrained TabPFN, frozen)
        │       ├─ z_query (B, tabpfn_dim)   ← pre-MLP hidden state
        │       └─ z_nbr   (B, k, tabpfn_dim)
        │
        ├─ node_proj: concat(z, e_loc) → GELU → h  (B, node_dim)
        │   (shared weights for query and neighbour nodes)
        │
        └─ GWRContextModule  (k-agnostic, end-to-end)
                ├─ y-inject: h_nbr + f(y_nbr) → h_aug (B, k, H)
                ├─ attn(h_query, h_aug, dist) → w (B, k)
                ├─ context = Σ w_j · h_aug_j   (B, H)
                ├─ β_i = beta_head(h_query + context)  (B, E)
                └─ ŷ_i = (z_query @ W_static) · β_i   (B,)

Design properties
-----------------
• k is a pure hyperparameter — no z_proj_dim ≤ k/2 constraint
• y_nbr is directly injected into neighbour representations, giving the
  spatial attention weights (and hence β) full access to local label info
• β_i ∈ R^E is a spatially-varying local coefficient vector, interpretable
  on the map (GWR spirit preserved)
• Trainable layers only on top of frozen base models (GeoCLIP, TabPFN)

Trainable parameters
--------------------
  loc_enc.proj         : 512 → loc_proj_dim
  node_proj            : (tabpfn_dim + loc_proj_dim) → node_dim
  ctx_mod.y_proj       : 1 → y_inject_dim
  ctx_mod.y_inject     : y_inject_dim → node_dim
  ctx_mod.query_head   : node_dim → attn_dim
  ctx_mod.key_head     : node_dim → attn_dim
  ctx_mod.W_static     : tabpfn_dim × z_proj_dim
  ctx_mod.beta_head    : node_dim → z_proj_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders       import LocationFusion
from .kernel         import GWRContextModule
from .tabpfn_encoder import TabPFNInContextEncoder


class GWF(nn.Module):
    """
    Geographical Weights Foundation Model (v2).

    Parameters
    ----------
    feat_dim      : number of raw tabular features (p)
    loc_proj_dim  : GeoCLIP projection output dim   (default 64)
    node_dim      : node representation dim         (default 128)
    z_proj_dim    : β and z-projection dim (E)      (default 32)
                    No constraint on k — freely tunable.
    attn_dim      : query/key dim for attention     (default 64)
    y_inject_dim  : embedding dim for y_nbr injection (default 8)
    tabpfn_path   : local TabPFN .ckpt; None → HuggingFace download
    """

    def __init__(
        self,
        feat_dim:     int        = 8,
        loc_proj_dim: int        = 64,
        node_dim:     int        = 128,
        z_proj_dim:   int        = 32,
        attn_dim:     int        = 64,
        y_inject_dim: int        = 8,
        tabpfn_path:  str | None = None,
    ):
        super().__init__()

        self.feat_dim   = feat_dim
        self.z_proj_dim = z_proj_dim

        # ── Frozen base encoders ──────────────────────────────────────────
        self.loc_enc = LocationFusion(geo_dim=512, out_dim=loc_proj_dim)
        self.ctx_enc = TabPFNInContextEncoder(model_path=tabpfn_path)
        tabpfn_dim   = self.ctx_enc.emb_dim             # e.g. 192

        # ── Shared node projection ────────────────────────────────────────
        self.node_proj = nn.Linear(tabpfn_dim + loc_proj_dim, node_dim,
                                   bias=True)

        # ── Context module (y-injection + attention + β generation) ───────
        self.ctx_mod = GWRContextModule(
            node_dim     = node_dim,
            tabpfn_dim   = tabpfn_dim,
            z_proj_dim   = z_proj_dim,
            attn_dim     = attn_dim,
            y_inject_dim = y_inject_dim,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _project_node(self, z, e_loc):
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
        beta  : (B, z_proj_dim) — local GWR coefficients (spatially varying)
        """
        coord     = batch["coord"]
        x         = batch["x"]
        nbr_coord = batch["nbr_coord"]
        nbr_x     = batch["nbr_x"]
        nbr_y     = batch["nbr_y"]
        nbr_dist  = batch["nbr_dist"]

        B, k, p = nbr_x.shape

        # 1. TabPFN in-context embeddings
        z_query, z_nbr = self.ctx_enc(x, nbr_x, nbr_y)    # (B,D), (B,k,D)

        # 2. GeoCLIP location embeddings
        e_loc_query = self.loc_enc(coord)                   # (B, L)
        e_loc_nbr   = self.loc_enc(nbr_coord)               # (B, k, L)

        # 3. Node projection (shared weights for query and neighbours)
        h_query = self._project_node(z_query, e_loc_query)  # (B, H)
        h_nbr   = self._project_node(
            z_nbr.reshape(B * k, -1),
            e_loc_nbr.reshape(B * k, -1),
        ).reshape(B, k, -1)                                  # (B, k, H)

        # 4. Context module: y-injection → attention → β → prediction
        y_hat, beta, _ = self.ctx_mod(
            h_query, h_nbr, nbr_y, z_query, dist=nbr_dist)

        return y_hat, beta

    # ─────────────────────────────────────────────────────────────────────────

    def loss(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_hat, y_true)

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_with_betas(self, dataloader, device: str = "cpu"):
        """
        Full-dataset inference. Returns predictions, β vectors, coords, labels.

        Returns
        -------
        y_hat  : (N,)
        betas  : (N, z_proj_dim)  — spatially-varying GWR-style coefficients
        coords : (N, 2)
        y_true : (N,)
        """
        self.eval()
        all_yhat, all_beta, all_coord, all_y = [], [], [], []
        for batch in dataloader:
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            yh, beta = self(batch)
            all_yhat.append(yh.cpu())
            all_beta.append(beta.cpu())
            all_coord.append(batch["coord"].cpu())
            all_y.append(batch["y"].cpu())

        return (torch.cat(all_yhat),
                torch.cat(all_beta),
                torch.cat(all_coord),
                torch.cat(all_y))
