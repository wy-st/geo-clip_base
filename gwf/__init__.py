"""
GWF — Geographical Weights Foundation Model
============================================

A foundation-model approach to Geographically Weighted Regression (GWR)
that combines:

  • GeoCLIP   — pretrained geographic location encoder (frozen)
  • SatCLIP   — satellite-scale location encoder (frozen)
  • TabPFN    — pretrained in-context encoder; pre-MLP hidden states used
                as context-aware tabular embeddings (frozen)
  • Dynamic kernel matrix K_i ∈ R^{p×p} — high-dimensional feature mixing
  • Closed-form matrix WLS — explicit, interpretable local regression

Key properties
--------------
  • β_i ∈ R^p per query point — high-dimensional local coefficients,
    visualisable on the map via t-SNE (no scalar bottleneck)
  • Minimal trainable layers — only single linear layers sit on top of
    frozen base models, preserving their learned representations
  • Differentiable WLS — end-to-end trainable via torch.linalg.solve

Quick start
-----------
    from gwf import GWF, GWFConfig, get_dataloaders

    cfg = GWFConfig(tabpfn_path="/path/to/tabpfn-v2-regressor.ckpt")
    train_dl, val_dl, feat_dim = get_dataloaders("synthetic", k=cfg.k_neighbors)
    model = GWF(feat_dim=feat_dim, tabpfn_path=cfg.tabpfn_path)
"""

from .config          import GWFConfig
from .data            import get_dataloaders, SpatialRegressionDataset, load_custom_dataset
from .encoders        import GeoCLIPEncoder, SatCLIPEncoder, LocationFusion
from .kernel          import DynamicKernelGenerator
from .regression      import MatrixGWR
from .tabpfn_encoder  import TabPFNInContextEncoder
from .model           import GWF

__all__ = [
    "GWFConfig",
    "get_dataloaders",
    "load_custom_dataset",
    "SpatialRegressionDataset",
    "GeoCLIPEncoder",
    "SatCLIPEncoder",
    "LocationFusion",
    "DynamicKernelGenerator",
    "MatrixGWR",
    "TabPFNInContextEncoder",
    "GWF",
]
