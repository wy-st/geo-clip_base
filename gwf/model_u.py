"""
GWF-U — Geographical Weights Foundation Model with Uncertainty

Extends GWF with Bayesian posterior uncertainty estimation.
Architecture is identical to GWF; only the WLS regression step is replaced by
UncertainMatrixGWR, which additionally returns a predictive standard deviation σ.

Uncertainty scheme (zero extra parameters):
──────────────────────────────────────────
The ridge WLS solution admits a Bayesian interpretation:

  β̂   = A^{-1} b            (MAP estimate; same as GWF)
  A    = Z̃^T diag(w) Z̃ + λI  (regularised normal matrix)

Local observation noise estimated from weighted residuals:
  σ²  = Σ_j w_j (y_j - z̃_j · β̂)²

Posterior predictive variance (propagates coefficient uncertainty to ŷ):
  Var(ŷ) = σ² (z̃_q^T A^{-1} z̃_q + 1)
            ╰─────────────────────╯   ╰─╯
            coefficient uncertainty   noise

Two sources of uncertainty are naturally separated:
  • z̃_q^T A^{-1} z̃_q : extrapolation / data-sparsity (high when z_query
                         is far from the support of the local neighbourhood)
  • σ²               : irreducible noise in the neighbourhood labels

Training: Gaussian NLL loss instead of MSE.
  L = ½ [(ŷ - y)² / Var(ŷ) + log Var(ŷ)]

Forward returns (y_hat, sigma, beta):
  y_hat : (B,)            — point predictions
  sigma : (B,)            — predictive standard deviations
  beta  : (B, z_proj_dim) — local MAP coefficients (same as GWF.beta)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders       import LocationFusion
from .kernel         import DynamicKernelGenerator
from .regression_u   import UncertainMatrixGWR, gaussian_nll_loss
from .tabpfn_encoder import TabPFNInContextEncoder


# ─────────────────────────────────────────────────────────────────────────────
# GWF with Uncertainty
# ─────────────────────────────────────────────────────────────────────────────

class GWF_U(nn.Module):
    """
    GWF with Bayesian posterior uncertainty estimation.

    Parameters (identical to GWF, plus nll_beta_weight)
    ----------
    feat_dim        : number of raw tabular features  (p)
    loc_proj_dim    : output dim of LocationFusion
    node_dim        : node representation dimension
    z_proj_dim      : regression dim in TabPFN embedding space  (β dimension)
    kernel_rank     : low-rank factor r for K_z = U @ V^T
    attn_dim        : query/key dim for spatial attention weights
    wls_lambda      : ridge regularisation in WLS
    nll_beta_weight : weight for σ² regularisation term in NLL loss
                      (prevents σ² from collapsing to near-zero)
    tabpfn_path     : path to a local TabPFN regressor .ckpt;
                      None → HuggingFace download
    """

    def __init__(
        self,
        feat_dim:         int   = 8,
        loc_proj_dim:     int   = 256,
        node_dim:         int   = 256,
        z_proj_dim:       int   = 64,
        kernel_rank:      int   = 4,
        attn_dim:         int   = 64,
        wls_lambda:       float = 1e-3,
        nll_beta_weight:  float = 0.0,
        tabpfn_path:      str | None = None,
    ):
        super().__init__()

        self.feat_dim        = feat_dim
        self.z_proj_dim      = z_proj_dim
        self.nll_beta_weight = nll_beta_weight

        # ── Location encoder (GeoCLIP + SatCLIP, frozen) ──────────────────
        self.loc_enc = LocationFusion(
            geo_dim=512, sat_dim=512, out_dim=loc_proj_dim)

        # ── TabPFN in-context encoder (pretrained, frozen) ─────────────────
        self.ctx_enc = TabPFNInContextEncoder(model_path=tabpfn_path)
        tabpfn_dim = self.ctx_enc.emb_dim

        # ── Node projection: [tabpfn_emb ‖ loc_proj] → node_dim ───────────
        self.node_proj = nn.Linear(tabpfn_dim + loc_proj_dim, node_dim,
                                   bias=True)

        # ── Dynamic kernel matrix + spatial attention (GNNWR-style) ───────
        self.kernel_gen = DynamicKernelGenerator(
            node_dim=node_dim,
            tabpfn_dim=tabpfn_dim,
            z_proj_dim=z_proj_dim,
            rank=kernel_rank,
            attn_dim=attn_dim)

        # ── WLS with Bayesian posterior uncertainty (no extra parameters) ──
        self.gwr = UncertainMatrixGWR(lam=wls_lambda)

    # ─────────────────────────────────────────────────────────────────────────

    def _project_node(self, z, e_loc):
        return F.gelu(self.node_proj(torch.cat([z, e_loc], dim=-1)))

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch: dict
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch : dict with keys
            coord, x, nbr_coord, nbr_x, nbr_y, nbr_dist

        Returns
        -------
        y_hat : (B,)            — point predictions
        sigma : (B,)            — predictive standard deviations
        beta  : (B, z_proj_dim) — local MAP coefficients
        """
        coord     = batch["coord"]
        x         = batch["x"]
        nbr_coord = batch["nbr_coord"]
        nbr_x     = batch["nbr_x"]
        nbr_y     = batch["nbr_y"]
        nbr_dist  = batch["nbr_dist"]

        B, k, p = nbr_x.shape

        # ── 1. TabPFN: context-aware embeddings ────────────────────────────
        z_query, z_nbr = self.ctx_enc(x, nbr_x, nbr_y)

        # ── 2. Location embeddings ─────────────────────────────────────────
        e_loc_query = self.loc_enc(coord)
        e_loc_nbr   = self.loc_enc(nbr_coord)

        # ── 3. Node projection ─────────────────────────────────────────────
        h_query = self._project_node(z_query, e_loc_query)
        z_nbr_flat  = z_nbr.reshape(B * k, -1)
        e_loc_nbr_f = e_loc_nbr.reshape(B * k, -1)
        h_nbr = self._project_node(z_nbr_flat, e_loc_nbr_f).reshape(B, k, -1)

        # ── 4. K_z + spatial attention weights (GNNWR-style) ──────────────
        K_z = self.kernel_gen.get_z_kernel_matrix(h_query)
        w   = self.kernel_gen.get_attention_weights(h_query, h_nbr, dist=nbr_dist)

        # ── 5. WLS with posterior uncertainty ─────────────────────────────
        y_hat, sigma, beta = self.gwr(z_query, K_z, z_nbr, nbr_y, w)

        return y_hat, sigma, beta

    # ─────────────────────────────────────────────────────────────────────────

    def loss(self, y_hat: torch.Tensor, sigma: torch.Tensor,
             y_true: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL loss (use instead of MSE during training)."""
        return gaussian_nll_loss(y_hat, sigma, y_true,
                                 beta_weight=self.nll_beta_weight)

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_with_uncertainty(self, dataloader, device: str = "cpu"):
        """
        Run inference, collecting predictions, uncertainties, and coefficients.

        Returns
        -------
        y_hat  : (N,)            — point predictions
        sigma  : (N,)            — predictive standard deviations
        betas  : (N, z_proj_dim) — local MAP coefficients
        coords : (N, 2)          — query coordinates
        y_true : (N,)            — ground-truth targets
        """
        self.eval()
        all_yhat, all_sigma, all_beta, all_coord, all_y = [], [], [], [], []
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            yh, sig, beta = self(batch)
            all_yhat.append(yh.cpu())
            all_sigma.append(sig.cpu())
            all_beta.append(beta.cpu())
            all_coord.append(batch["coord"].cpu())
            all_y.append(batch["y"].cpu())

        return (torch.cat(all_yhat),
                torch.cat(all_sigma),
                torch.cat(all_beta),
                torch.cat(all_coord),
                torch.cat(all_y))
