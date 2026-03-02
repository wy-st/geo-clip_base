"""
GWF-U — GWF with BNN-style Uncertainty

Architecture is identical to GWF-v2.  Adds one learnable log_sigma that
scales a noise perturbation on β, supporting MC-dropout-style uncertainty.

Training
--------
    eps = torch.randn(B, model.z_proj_dim, device=device)
    y_hat, _ = model(batch, eps_beta=eps)
    loss = model.loss(y_hat, batch["y"])

Inference
---------
• MAP point estimate  : model(batch)                    (eps_beta = None)
• Predictive σ        : model.predict_with_uncertainty() (MC sampling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders       import LocationFusion
from .kernel         import GWRContextModule
from .tabpfn_encoder import TabPFNInContextEncoder


class GWF_U(nn.Module):
    """
    Parameters
    ----------
    feat_dim      : number of raw tabular features
    loc_proj_dim  : GeoCLIP projection output dim   (default 64)
    node_dim      : node representation dim         (default 128)
    z_proj_dim    : β and z-projection dim (E)      (default 32)
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

        self.loc_enc = LocationFusion(geo_dim=512, out_dim=loc_proj_dim)
        self.ctx_enc = TabPFNInContextEncoder(model_path=tabpfn_path)
        tabpfn_dim   = self.ctx_enc.emb_dim

        self.node_proj = nn.Linear(tabpfn_dim + loc_proj_dim, node_dim, bias=True)

        self.ctx_mod = GWRContextModule(
            node_dim     = node_dim,
            tabpfn_dim   = tabpfn_dim,
            z_proj_dim   = z_proj_dim,
            attn_dim     = attn_dim,
            y_inject_dim = y_inject_dim,
        )

        # Learnable noise scale for BNN reparameterisation
        self.log_sigma = nn.Parameter(torch.zeros(1))

    # ─────────────────────────────────────────────────────────────────────────

    def _project_node(self, z, e_loc):
        return F.gelu(self.node_proj(torch.cat([z, e_loc], dim=-1)))

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self,
                batch:    dict,
                eps_beta: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        coord     = batch["coord"]
        x         = batch["x"]
        nbr_coord = batch["nbr_coord"]
        nbr_x     = batch["nbr_x"]
        nbr_y     = batch["nbr_y"]
        nbr_dist  = batch["nbr_dist"]

        B, k, _ = nbr_x.shape

        z_query, z_nbr = self.ctx_enc(x, nbr_x, nbr_y)

        e_loc_query = self.loc_enc(coord)
        e_loc_nbr   = self.loc_enc(nbr_coord)

        h_query = self._project_node(z_query, e_loc_query)
        h_nbr   = self._project_node(
            z_nbr.reshape(B * k, -1),
            e_loc_nbr.reshape(B * k, -1),
        ).reshape(B, k, -1)

        y_hat, beta, _ = self.ctx_mod(
            h_query, h_nbr, nbr_y, z_query, dist=nbr_dist)

        if eps_beta is not None:
            # Reparameterisation: perturb β by learnable σ
            sigma  = self.log_sigma.exp()
            z_proj = z_query @ self.ctx_mod.W_static     # (B, E)
            y_hat  = y_hat + (z_proj * (sigma * eps_beta)).sum(dim=-1)

        return y_hat, beta

    # ─────────────────────────────────────────────────────────────────────────

    def loss(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_hat, y_true)

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_with_uncertainty(self,
                                 dataloader,
                                 device: str = "cpu",
                                 n_mc:   int = 50,
                                 ) -> tuple:
        self.eval()
        e = self.z_proj_dim
        all_yhat, all_sigma, all_beta, all_coord, all_y = [], [], [], [], []

        for batch in dataloader:
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            B = batch["coord"].shape[0]

            yh_map, beta_map = self(batch)

            preds = []
            for _ in range(n_mc):
                eps = torch.randn(B, e, device=device)
                yh_i, _ = self(batch, eps_beta=eps)
                preds.append(yh_i)
            sigma = torch.stack(preds).std(dim=0)

            all_yhat.append(yh_map.cpu())
            all_sigma.append(sigma.cpu())
            all_beta.append(beta_map.cpu())
            all_coord.append(batch["coord"].cpu())
            all_y.append(batch["y"].cpu())

        return (
            torch.cat(all_yhat),
            torch.cat(all_sigma),
            torch.cat(all_beta),
            torch.cat(all_coord),
            torch.cat(all_y),
        )
