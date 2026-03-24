"""
gwf/data/dataset.py
====================
GWFDataset: a flexible dataset for spatial regression / classification.

Supports loading from:
  - GeoJSON FeatureCollection  (source="geojson")
  - CSV with lat/lon columns   (source="csv")
  - In-memory pandas DataFrame (source="dataframe")

Each item in the dataset:
  coords     : (2,)  latitude, longitude
  X_tab      : (p,)  normalised tabular features
  y          : ()    target label
  image      : (C,H,W) optional — if images_dir is provided
  subgraph   : PyG Data optional — if subgraphs_dir is provided

The prompt_text is stored as a dataset-level attribute (shared across all
points) and passed to the model's encoder_bank.llm.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class GWFDataset(Dataset):
    """
    Spatial regression/classification dataset.

    After construction, all numeric data is stored as float32 tensors.
    Optional image and subgraph data can be added later via
    set_images() and set_subgraphs().

    Args:
        coords      : (N, 2) latitude/longitude  float32
        X_tab       : (N, p) tabular features    float32 (should be pre-normalised)
        y           : (N,)   targets             float32
        prompt_text : dataset-level text description for LLM channel
        images      : (N, C, H, W) optional satellite images
        subgraphs   : list[PyG Data] optional POI/road subgraphs
    """

    def __init__(
        self,
        coords:      torch.Tensor,
        X_tab:       torch.Tensor,
        y:           torch.Tensor,
        prompt_text: str                       = "",
        images:      Optional[torch.Tensor]    = None,
        subgraphs                              = None,
    ):
        super().__init__()
        N = coords.shape[0]
        assert X_tab.shape[0] == N and y.shape[0] == N, \
            "coords, X_tab, y must all have the same number of rows."

        self.coords      = coords.float()
        self.X_tab       = X_tab.float()
        self.y           = y.float()
        self.prompt_text = prompt_text
        self.images      = images
        self.subgraphs   = subgraphs

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {
            "coords": self.coords[idx],    # (2,)
            "X_tab":  self.X_tab[idx],     # (p,)
            "y":      self.y[idx],         # ()
        }
        if self.images is not None:
            item["image"] = self.images[idx]       # (C, H, W)
        if self.subgraphs is not None:
            item["subgraph"] = self.subgraphs[idx]
        return item

    def set_images(self, images: torch.Tensor):
        """Attach satellite images after construction."""
        assert images.shape[0] == len(self), "images must have N rows."
        self.images = images.float()

    def set_subgraphs(self, subgraphs: list):
        """Attach POI/road subgraphs after construction."""
        assert len(subgraphs) == len(self), "subgraphs must have N entries."
        self.subgraphs = subgraphs


# =============================================================================
# Factory functions: load from different sources
# =============================================================================

def _read_geojson(path: str, target_col: str, feature_cols: list[str]) -> tuple:
    """Load a GeoJSON FeatureCollection into numpy arrays."""
    with open(path, "r") as f:
        gj = json.load(f)
    features = gj["features"]

    lats, lons, ys, Xs = [], [], [], []
    for feat in features:
        props = feat["properties"]
        geom  = feat["geometry"]

        # Geometry: Point expected
        if geom["type"] == "Point":
            lon, lat = geom["coordinates"][:2]
        else:
            # Use centroid for non-point geometries
            coords_raw = np.array(geom["coordinates"])
            lon = float(coords_raw[:, 0].mean())
            lat = float(coords_raw[:, 1].mean())

        if props.get(target_col) is None:
            continue

        lats.append(lat)
        lons.append(lon)
        ys.append(float(props[target_col]))
        Xs.append([float(props.get(c, 0.0)) for c in feature_cols])

    coords_np = np.column_stack([lats, lons]).astype(np.float32)
    X_np      = np.array(Xs, dtype=np.float32)
    y_np      = np.array(ys, dtype=np.float32)
    return coords_np, X_np, y_np


def _read_csv(path: str, target_col: str, feature_cols: list[str],
              lat_col: str = "lat", lon_col: str = "lon") -> tuple:
    """Load a CSV file into numpy arrays."""
    import pandas as pd
    df = pd.read_csv(path)
    lats = df[lat_col].values.astype(np.float32)
    lons = df[lon_col].values.astype(np.float32)
    coords_np = np.column_stack([lats, lons])
    X_np      = df[feature_cols].values.astype(np.float32)
    y_np      = df[target_col].values.astype(np.float32)
    return coords_np, X_np, y_np


def _normalise(X_np: np.ndarray) -> np.ndarray:
    """Standard-normalise each feature column. Clips extreme values."""
    mu  = X_np.mean(axis=0, keepdims=True)
    std = X_np.std(axis=0, keepdims=True).clip(1e-8)
    return (X_np - mu) / std


def load_gwf_dataset(
    cfg:         dict,
    prompt_text: str = "",
) -> "GWFDataset":
    """
    Create a GWFDataset from a config dict.

    Config keys:
        source       : "geojson" | "csv" | required
        path         : file path
        target_col   : name of target column
        feature_cols : list of feature column names
        lat_col      : (csv only) latitude column name
        lon_col      : (csv only) longitude column name
        normalise    : bool, whether to z-normalise X_tab (default True)
    """
    source      = cfg["source"]
    path        = cfg["path"]
    target_col  = cfg["target_col"]
    feature_cols = cfg["feature_cols"]

    if source == "geojson":
        coords_np, X_np, y_np = _read_geojson(path, target_col, feature_cols)
    elif source == "csv":
        lat_col = cfg.get("lat_col", "lat")
        lon_col = cfg.get("lon_col", "lon")
        coords_np, X_np, y_np = _read_csv(path, target_col, feature_cols,
                                           lat_col, lon_col)
    else:
        raise ValueError(f"Unknown data source: {source!r}. Use 'geojson' or 'csv'.")

    if cfg.get("normalise", True):
        X_np = _normalise(X_np)

    coords = torch.from_numpy(coords_np)
    X_tab  = torch.from_numpy(X_np)
    y      = torch.from_numpy(y_np)

    return GWFDataset(coords, X_tab, y, prompt_text=prompt_text)


def get_dataloaders(
    dataset: "GWFDataset",
    val_split:  float = 0.2,
    batch_size: int   = 256,
    seed:       int   = 42,
    num_workers: int  = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Split dataset into train/val and return DataLoaders.

    Returns:
        train_loader, val_loader
    """
    N   = len(dataset)
    n_val   = int(N * val_split)
    n_train = N - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
