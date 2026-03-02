"""
GWF — Geographical Weights Foundation Model  (v2)
==================================================

Combines:
  • GeoCLIP  — pretrained geographic location encoder (frozen)
  • TabPFN   — pretrained in-context encoder (frozen)
  • GWRContextModule — y-injection + spatial attention + context-driven β

Key properties
--------------
  • β_i ∈ R^E per query point — spatially-varying GWR-style coefficients,
    visualisable on the map via PCA / UMAP
  • k is a pure hyperparameter — no z_proj_dim ≤ k constraint
  • y_nbr injected into neighbour representations; β shaped by local labels

Quick start
-----------
    from gwf import GWF, GWFConfig, get_dataloaders

    cfg = GWFConfig(tabpfn_path="/path/to/tabpfn-v2-regressor.ckpt")
    train_dl, val_dl, feat_dim = get_dataloaders("synthetic", k=cfg.k_neighbors)
    model = GWF(feat_dim=feat_dim, tabpfn_path=cfg.tabpfn_path)
"""

from .config          import GWFConfig
from .data            import (get_dataloaders, SpatialRegressionDataset,
                               load_custom_dataset, load_geojson_dataset)
from .encoders        import GeoCLIPEncoder, LocationFusion
from .kernel          import GWRContextModule
from .regression      import MatrixGWR
from .tabpfn_encoder  import TabPFNInContextEncoder
from .model           import GWF

__all__ = [
    "GWFConfig",
    "get_dataloaders",
    "load_custom_dataset",
    "load_geojson_dataset",
    "SpatialRegressionDataset",
    "GeoCLIPEncoder",
    "LocationFusion",
    "GWRContextModule",
    "MatrixGWR",
    "TabPFNInContextEncoder",
    "GWF",
]
