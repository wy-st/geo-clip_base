"""
GWF — Geographical Weights Foundation Model
============================================

A foundation-model approach to Geographically Weighted Regression (GWR)
that combines:

  • GeoCLIP   — pretrained geographic location encoder (frozen)
  • SatCLIP   — satellite-scale location encoder (frozen)
  • TabPFN-inspired in-context encoder — neighbourhood-aware tabular features
  • Dynamic kernel matrix K_i ∈ R^{p×p} — high-dimensional feature mixing
  • Closed-form matrix WLS — explicit, interpretable local regression

Key properties
--------------
  • β_i ∈ R^p per query point — high-dimensional local coefficients,
    visualisable on the map via t-SNE (no scalar bottleneck)
  • Minimal MLP layers — only single linear layers sit on top of
    frozen base models, preserving their learned representations
  • Differentiable WLS — end-to-end trainable via torch.linalg.solve

Quick start
-----------
    from gwf import GWF, GWFConfig, get_dataloaders

    cfg = GWFConfig()
    train_dl, val_dl, feat_dim = get_dataloaders("synthetic", k=cfg.k_neighbors)
    model = GWF(feat_dim=feat_dim)
"""

from .config     import GWFConfig
from .data       import get_dataloaders, SpatialRegressionDataset
from .encoders   import GeoCLIPEncoder, SatCLIPEncoder, LocationFusion
from .kernel     import DynamicKernelGenerator
from .regression import MatrixGWR
from .model      import GWF, InContextEncoder

__all__ = [
    "GWFConfig",
    "get_dataloaders",
    "SpatialRegressionDataset",
    "GeoCLIPEncoder",
    "SatCLIPEncoder",
    "LocationFusion",
    "DynamicKernelGenerator",
    "MatrixGWR",
    "GWF",
    "InContextEncoder",
]
