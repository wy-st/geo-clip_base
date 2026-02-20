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

    # ── Tabular features ───────────────────────────────────────────────────
    feat_dim: int = 8              # number of raw tabular features (set by data)
    feat_emb_dim: int = 64         # in-context cross-attention output dim

    # ── Node representation ────────────────────────────────────────────────
    node_dim: int = 256            # concat(feat_emb, loc_proj) → 1 linear → node_dim

    # ── Dynamic kernel matrix (low-rank, p×p) ──────────────────────────────
    # K_i = U_i @ V_i^T,  U_i,V_i ∈ R^{p×rank_k}
    kernel_rank: int = 4           # low-rank for K_i  (p=feat_dim, rank << p)

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
