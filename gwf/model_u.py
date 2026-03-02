"""
GWF-U — GWF with BNN-style Uncertainty

Architecture is identical to GWF.  The only difference is that the WLS step
uses UncertainMatrixGWR, which adds one learnable scalar log_sigma and
supports reparameterized β sampling at training time.

Training
--------
Sample ε from N(0, I) and pass it as eps_beta so that gradients flow through
the sampled β via the reparameterization trick.  The loss is plain MSE:

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
from .kernel         import DynamicKernelGenerator
from .regression_u   import UncertainMatrixGWR
from .tabpfn_encoder import TabPFNInContextEncoder


class GWF_U(nn.Module):
    """
    Parameters  (identical to GWF, minus nll_beta_weight)
    ----------
    feat_dim        : number of raw tabular features
    loc_proj_dim    : output dim of LocationFusion
    node_dim        : node representation dimension
    z_proj_dim      : regression dim in TabPFN embedding space
    kernel_rank     : low-rank factor r for K_z = U @ V^T
    attn_dim        : query/key dim for spatial attention weights
    wls_lambda      : ridge regularisation in WLS
    tabpfn_path     : path to a local TabPFN regressor .ckpt;
                      None → HuggingFace download
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

        self.loc_enc = LocationFusion(
            geo_dim=512, sat_dim=512, out_dim=loc_proj_dim,
            satclip_ckpt=satclip_ckpt)

        self.ctx_enc = TabPFNInContextEncoder(model_path=tabpfn_path)
        tabpfn_dim = self.ctx_enc.emb_dim

        self.node_proj = nn.Linear(tabpfn_dim + loc_proj_dim, node_dim, bias=True)

        self.kernel_gen = DynamicKernelGenerator(
            node_dim=node_dim,
            tabpfn_dim=tabpfn_dim,
            z_proj_dim=z_proj_dim,
            rank=kernel_rank,
            attn_dim=attn_dim)

        # UncertainMatrixGWR adds one learnable log_sigma over plain GWF
        self.gwr = UncertainMatrixGWR(lam=wls_lambda)

    # ─────────────────────────────────────────────────────────────────────────

    def _project_node(self, z, e_loc):
        return F.gelu(self.node_proj(torch.cat([z, e_loc], dim=-1)))

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self,
                batch:    dict,
                eps_beta: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch    : dict with keys coord, x, nbr_coord, nbr_x, nbr_y, nbr_dist
        eps_beta : (B, z_proj_dim) | None
                   External noise for BNN reparameterization.
                   None → deterministic MAP prediction.

        Returns
        -------
        y_hat : (B,)           — point prediction
        beta  : (B, z_proj_dim) — MAP local coefficients μ_β
        """
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

        K_z = self.kernel_gen.get_z_kernel_matrix(h_query)
        w   = self.kernel_gen.get_attention_weights(h_query, h_nbr, dist=nbr_dist)

        y_hat, beta = self.gwr(z_query, K_z, z_nbr, nbr_y, w, eps_beta=eps_beta)

        return y_hat, beta

    # ─────────────────────────────────────────────────────────────────────────

    def loss(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Plain MSE — no distributional assumption."""
        return F.mse_loss(y_hat, y_true)

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_with_uncertainty(self,
                                 dataloader,
                                 device: str = "cpu",
                                 n_mc:   int = 50,
                                 ) -> tuple:
        """
        Monte-Carlo inference.

        For each sample run a stochastic forward pass with ε ~ N(0, I);
        the standard deviation across samples is the predictive uncertainty.
        The point estimate is the deterministic MAP (ε = 0).

        Returns
        -------
        y_hat  : (N,)            — MAP point predictions
        sigma  : (N,)            — predictive std  (MC estimate)
        betas  : (N, z_proj_dim) — MAP local coefficients
        coords : (N, 2)
        y_true : (N,)
        """
        self.eval()
        e = self.z_proj_dim
        all_yhat, all_sigma, all_beta, all_coord, all_y = [], [], [], [], []

        for batch in dataloader:
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            B = batch["coord"].shape[0]

            # MAP point estimate
            yh_map, beta_map = self(batch)

            # MC samples for σ
            preds = []
            for _ in range(n_mc):
                eps = torch.randn(B, e, device=device)
                yh_i, _ = self(batch, eps_beta=eps)
                preds.append(yh_i)
            sigma = torch.stack(preds).std(dim=0)   # (B,)

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
