"""
GWF-U — Geographical Weights Foundation Model with BNN-style Uncertainty

Extends GWF with two independent sources of uncertainty, both modelled via
the reparameterization trick and estimated by Monte-Carlo sampling:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source 1 — β coefficient uncertainty
  The WLS posterior β | y ~ N(μ_β, σ²_obs A⁻¹) is sampled via Cholesky
  reparameterization in UncertainMatrixGWR:
      β = μ_β + σ_obs · L⁻ᵀ ε_β,   ε_β ~ N(0, I_e)
  This is the Bayesian WLS posterior expressed as a BNN layer — no extra
  learnable parameters; μ_β and σ_obs are derived analytically from the
  local data geometry.

Source 2 — Spatial neighbourhood-range uncertainty
  DynamicKernelGenerator perturbs the attention logits before softmax:
      scores_noisy = scores + exp(log_sigma_attn) · ε_attn,  ε_attn ~ N(0, I_k)
  log_sigma_attn is a single learnable scalar.  Different ε_attn give
  different spatial-weight vectors w → different WLS solutions → different ŷ.
  Large σ_spatial in prediction means the target's value is sensitive to
  which neighbours are included — i.e. it sits near a spatial boundary or
  transition zone.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training — pure MSE (no distributional assumption)
  • Single stochastic forward pass per batch:
      y_hat_sample = model(batch, eps_attn, eps_beta)
      loss = MSE(y_hat_sample, y_true)
  • Gradients flow through both noise paths via reparameterization.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Inference — uncertainty decomposition (law of total variance)
  predict_with_uncertainty() returns four uncertainty signals:

  σ_total    = std over (ε_attn, ε_β)             — total predictive σ
  σ_β        = std over ε_β  (ε_attn = 0)         — coefficient uncertainty
  σ_spatial  = sqrt(σ²_total − σ²_β).clamp(0)    — spatial-range uncertainty
  σ_obs      = sqrt(Σ_j w_j r²_j)                 — local aleatoric noise

  σ²_total ≈ σ²_β + σ²_spatial  (by approximate variance decomposition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders       import LocationFusion
from .kernel         import DynamicKernelGenerator
from .regression_u   import UncertainMatrixGWR
from .tabpfn_encoder import TabPFNInContextEncoder


# ─────────────────────────────────────────────────────────────────────────────
# GWF with BNN-style Uncertainty
# ─────────────────────────────────────────────────────────────────────────────

class GWF_U(nn.Module):
    """
    GWF with BNN-style reparameterized uncertainty.

    Parameters
    ----------
    feat_dim        : number of raw tabular features  (p)
    loc_proj_dim    : output dim of LocationFusion
    node_dim        : node representation dimension
    z_proj_dim      : regression dim in TabPFN embedding space  (β dimension)
    kernel_rank     : low-rank factor r for K_z = U @ V^T
    attn_dim        : query/key dim for spatial attention weights
    wls_lambda      : ridge regularisation in WLS
    tabpfn_path     : path to a local TabPFN regressor .ckpt;
                      None → HuggingFace download
    """

    def __init__(
        self,
        feat_dim:     int   = 8,
        loc_proj_dim: int   = 256,
        node_dim:     int   = 256,
        z_proj_dim:   int   = 64,
        kernel_rank:  int   = 4,
        attn_dim:     int   = 64,
        wls_lambda:   float = 1e-3,
        tabpfn_path:  str | None = None,
    ):
        super().__init__()

        self.feat_dim   = feat_dim
        self.z_proj_dim = z_proj_dim

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
        # DynamicKernelGenerator now also owns log_sigma_attn (spatial noise)
        self.kernel_gen = DynamicKernelGenerator(
            node_dim=node_dim,
            tabpfn_dim=tabpfn_dim,
            z_proj_dim=z_proj_dim,
            rank=kernel_rank,
            attn_dim=attn_dim)

        # ── WLS with BNN-style β reparameterization ────────────────────────
        self.gwr = UncertainMatrixGWR(lam=wls_lambda)

    # ─────────────────────────────────────────────────────────────────────────

    def _project_node(self, z, e_loc):
        return F.gelu(self.node_proj(torch.cat([z, e_loc], dim=-1)))

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self,
                batch:     dict,
                eps_attn:  torch.Tensor | None = None,
                eps_beta:  torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward pass.

        Parameters
        ----------
        batch    : dict with keys coord, x, nbr_coord, nbr_x, nbr_y, nbr_dist
        eps_attn : (B, k) | None  — spatial-range noise  (ε_attn ~ N(0, I_k))
        eps_beta : (B, e) | None  — β-coefficient noise  (ε_β   ~ N(0, I_e))

        When both eps are None: deterministic MAP forward (eval / comparison).
        During training: sample eps_attn and eps_beta externally and pass in.

        Returns
        -------
        y_hat     : (B,)  — point prediction
        sigma_obs : (B,)  — local WLS aleatoric noise  sqrt(Σ_j w_j r²_j)
        beta      : (B, e) — MAP coefficients μ_β  (always posterior mean)
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

        # ── 4. K_z + stochastic attention weights ─────────────────────────
        K_z = self.kernel_gen.get_z_kernel_matrix(h_query)
        w   = self.kernel_gen.get_attention_weights(
            h_query, h_nbr, dist=nbr_dist, eps=eps_attn)

        # ── 5. WLS with optional β reparameterization ─────────────────────
        y_hat, sigma_obs, beta = self.gwr(
            z_query, K_z, z_nbr, nbr_y, w, eps_beta=eps_beta)

        return y_hat, sigma_obs, beta

    # ─────────────────────────────────────────────────────────────────────────

    def loss(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Pure MSE loss for BNN-style training.

        Use with a stochastic forward pass:
            eps_a = torch.randn(B, k, device=device)
            eps_b = torch.randn(B, e, device=device)
            y_hat, _, _ = model(batch, eps_a, eps_b)
            l = model.loss(y_hat, batch["y"])

        No distributional assumption on y.  Gradients propagate through
        μ_β (prediction accuracy) and σ_obs (WLS fit quality) via
        both reparameterized noise paths.
        """
        return F.mse_loss(y_hat, y_true)

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_with_uncertainty(self,
                                 dataloader,
                                 device:  str = "cpu",
                                 n_mc:    int = 50,
                                 ) -> tuple:
        """
        MC inference with uncertainty decomposition.

        Runs n_mc stochastic forward passes per batch and decomposes the
        total predictive variance into two interpretable components via the
        law of total variance:

          σ²_total   = Var_{ε_attn, ε_β}(ŷ)          [all noise sources]
          σ²_β       = Var_{ε_β}(ŷ)  |  ε_attn=0     [β uncertainty only]
          σ²_spatial = (σ²_total − σ²_β).clamp(0)    [spatial-range contrib]

        Point estimate: deterministic MAP (both eps = None), so the mean
        prediction is unaffected by the stochastic noise calibration.

        Parameters
        ----------
        n_mc : number of MC samples for each variance estimate
               (n_mc forward passes are run twice per batch — for σ_total
               and σ_β — so total passes per batch = 1 + 2*n_mc)

        Returns
        -------
        y_hat      : (N,)  — MAP point predictions
        sigma_total    : (N,)  — total predictive std
        sigma_spatial  : (N,)  — spatial neighbourhood-range uncertainty std
        sigma_beta     : (N,)  — β coefficient uncertainty std
        sigma_obs      : (N,)  — local aleatoric noise (WLS residual std)
        betas          : (N, z_proj_dim)  — MAP local coefficients
        coords         : (N, 2)
        y_true         : (N,)
        """
        self.eval()
        e = self.z_proj_dim

        (all_yhat, all_s_total, all_s_spatial, all_s_beta,
         all_s_obs, all_beta, all_coord, all_y) = ([] for _ in range(8))

        for batch in dataloader:
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            B = batch["coord"].shape[0]
            k = batch["nbr_x"].shape[1]

            # ── MAP point prediction (deterministic) ──────────────────────
            yh_map, s_obs, beta_map = self(batch)

            # ── MC: total uncertainty (both ε_attn and ε_β) ───────────────
            mc_total = []
            for _ in range(n_mc):
                eps_a = torch.randn(B, k, device=device)
                eps_b = torch.randn(B, e, device=device)
                yh_i, _, _ = self(batch, eps_a, eps_b)
                mc_total.append(yh_i)
            sigma_total = torch.stack(mc_total).std(dim=0)   # (B,)

            # ── MC: β uncertainty only (ε_attn = 0, vary ε_β) ────────────
            mc_beta = []
            for _ in range(n_mc):
                eps_b = torch.randn(B, e, device=device)
                yh_i, _, _ = self(batch, None, eps_b)
                mc_beta.append(yh_i)
            sigma_beta = torch.stack(mc_beta).std(dim=0)     # (B,)

            # ── Spatial-range uncertainty (variance decomposition) ────────
            sigma_spatial = (
                sigma_total.pow(2) - sigma_beta.pow(2)
            ).clamp(min=0.0).sqrt()                           # (B,)

            all_yhat.append(yh_map.cpu())
            all_s_total.append(sigma_total.cpu())
            all_s_spatial.append(sigma_spatial.cpu())
            all_s_beta.append(sigma_beta.cpu())
            all_s_obs.append(s_obs.cpu())
            all_beta.append(beta_map.cpu())
            all_coord.append(batch["coord"].cpu())
            all_y.append(batch["y"].cpu())

        return (
            torch.cat(all_yhat),
            torch.cat(all_s_total),
            torch.cat(all_s_spatial),
            torch.cat(all_s_beta),
            torch.cat(all_s_obs),
            torch.cat(all_beta),
            torch.cat(all_coord),
            torch.cat(all_y),
        )
