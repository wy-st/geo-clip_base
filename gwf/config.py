"""
GWF (Geographical Weights Foundation Model) — Configuration
"""
from dataclasses import dataclass


@dataclass
class GWFConfig:
    # ── Spatial graph ──────────────────────────────────────────────────────
    k_neighbors: int = 16          # k-NN neighbours — pure hyperparameter,
                                   # not constrained by any model dimension

    # ── Location encoder (frozen GeoCLIP) ──────────────────────────────────
    geo_emb_dim:  int = 512        # GeoCLIP output dim (fixed)
    loc_proj_dim: int = 64         # after GeoCLIP linear projection

    # ── TabPFN in-context encoder (frozen pretrained transformer) ───────────
    tabpfn_path: str | None = None  # local .ckpt; None → HuggingFace download

    # ── Tabular features ───────────────────────────────────────────────────
    feat_dim: int = 8              # number of raw tabular features (set by data)

    # ── Node representation ────────────────────────────────────────────────
    node_dim: int = 128            # concat(tabpfn_emb, loc_proj) → Linear → H

    # ── z-space projection & β generation ─────────────────────────────────
    z_proj_dim: int = 32           # β and W_static dimension (E)
                                   # free to set; no k constraint
    y_inject_dim: int = 8          # small embedding dim for y_nbr injection

    # ── Attention weights ──────────────────────────────────────────────────
    attn_dim: int = 64             # query/key dimension for spatial attention

    # ── Training ───────────────────────────────────────────────────────────
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 512
    seed: int = 42

    # ── Data ───────────────────────────────────────────────────────────────
    n_samples: int = 5000          # synthetic dataset size (if used)
    val_split: float = 0.2
