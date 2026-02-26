"""
GWF (Geographical Weights Foundation Model) — Configuration
"""
from dataclasses import dataclass, field


@dataclass
class GWFConfig:
    # ── Spatial graph ──────────────────────────────────────────────────────
    k_neighbors: int = 16          # k-NN neighbours for each point
    coord_scale: float = 1.0       # scale applied to raw lat/lon before RFF

    # ── Location encoder (frozen GeoCLIP + frozen SatCLIP proxy) ───────────
    geo_emb_dim: int = 512         # GeoCLIP location-encoder output dim
    sat_emb_dim: int = 512         # SatCLIP-proxy output dim (same arch, diff sigma)
    loc_proj_dim: int = 256        # after fusing geo+sat  (1 linear layer)

    # ── TabPFN in-context encoder (frozen pretrained transformer) ───────────
    # feat_emb_dim is NOT set here; it is auto-detected from TabPFN's ninp
    # at model construction time and stored in GWF.ctx_enc.emb_dim.
    tabpfn_path: str | None = None  # local .ckpt path; None → HuggingFace download

    # ── Tabular features ───────────────────────────────────────────────────
    feat_dim: int = 8              # number of raw tabular features (set by data)

    # ── Node representation ────────────────────────────────────────────────
    # node_dim input = tabpfn_dim (auto) + loc_proj_dim (above)
    node_dim: int = 256            # concat(tabpfn_emb, loc_proj) → 1 linear → node_dim

    # ── z-space projection for WLS (GNNWR-style, TabPFN embedding space) ──
    # K_z = U_i @ V_i^T,  U_i ∈ R^{tabpfn_dim×rank},  V_i ∈ R^{z_proj_dim×rank}
    z_proj_dim: int = 64           # regression dim in TabPFN embedding space (β dim)
    kernel_rank: int = 4           # low-rank factor r for K_z = U @ V^T

    # ── Attention weights (query-key) ──────────────────────────────────────
    attn_dim: int = 64             # query/key dimension for spatial attention

    # ── WLS regularisation ─────────────────────────────────────────────────
    wls_lambda: float = 1e-3       # ridge term in (X̃ᵀWX̃ + λI)⁻¹

    # ── Training ───────────────────────────────────────────────────────────
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 512          # number of query points per step
    seed: int = 42

    # ── Data ───────────────────────────────────────────────────────────────
    n_samples: int = 5000          # synthetic dataset size (if used)
    val_split: float = 0.2
