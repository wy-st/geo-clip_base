"""
GWF — Data utilities

Two data sources:
  1. Synthetic: spatially structured random data (always available)
  2. California Housing: real-world house-price task with approximate coords
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic spatial dataset
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_spatial(n=5000, p=8, seed=42):
    """
    Generate spatially heterogeneous regression data.

    Spatial heterogeneity: the true β varies smoothly across the map,
    so local regression is clearly better than global OLS.

    Returns
    -------
    coords : ndarray (n, 2)  — [lat, lon] in realistic ranges
    X      : ndarray (n, p)  — standardised tabular features
    y      : ndarray (n,)    — continuous target
    """
    rng = np.random.default_rng(seed)

    # Scatter points across a roughly California-shaped bounding box
    lat = rng.uniform(32.5, 42.0, n)
    lon = rng.uniform(-124.5, -114.0, n)
    coords = np.stack([lat, lon], axis=1)

    # Raw tabular features
    X_raw = rng.standard_normal((n, p))

    # Spatially varying β: two smooth spatial functions of (lat, lon)
    lat_n = (lat - 37.0) / 5.0    # normalise around centre
    lon_n = (lon + 119.0) / 5.0

    beta_0 = np.sin(lat_n) * np.cos(lon_n)          # base intercept term
    beta_scale = 0.5 + 0.5 * np.tanh(lat_n + lon_n) # scale varies in [0,1]

    # Target: non-linear spatial heterogeneity + feature interaction
    y = (beta_0
         + beta_scale * X_raw[:, 0]
         + (1 - beta_scale) * X_raw[:, 1]
         + 0.3 * X_raw[:, 2] * X_raw[:, 3]          # feature interaction
         + 0.5 * lat_n ** 2
         + rng.normal(0, 0.15, n))                   # observation noise

    # Standardise features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)
    y = ((y - y.mean()) / y.std()).astype(np.float32)

    return coords.astype(np.float32), X, y


# ──────────────────────────────────────────────────────────────────────────────
# California Housing (optional — requires scikit-learn)
# ──────────────────────────────────────────────────────────────────────────────

def load_california_housing():
    """
    Load California Housing dataset.
    Coordinates are the actual lat/lon columns in the dataset.

    Returns same (coords, X, y) format as make_synthetic_spatial.
    """
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    df = data.data                         # (20640, 8)
    target = data.target                   # (20640,)
    feature_names = data.feature_names

    # Last two columns are Latitude and Longitude
    lat_idx = feature_names.index("Latitude")
    lon_idx = feature_names.index("Longitude")
    lat = df[:, lat_idx].astype(np.float32)
    lon = df[:, lon_idx].astype(np.float32)
    coords = np.stack([lat, lon], axis=1)

    # All 8 features as input (including lat/lon — the model can learn to
    # rely on the spatial encoder instead of raw coords)
    scaler = StandardScaler()
    X = scaler.fit_transform(df).astype(np.float32)

    y = target.astype(np.float32)
    y = ((y - y.mean()) / y.std()).astype(np.float32)

    return coords, X, y


# ──────────────────────────────────────────────────────────────────────────────
# k-NN graph (precomputed, stored with dataset)
# ──────────────────────────────────────────────────────────────────────────────

def build_knn_graph(coords: np.ndarray, k: int):
    """
    Build k-NN graph purely on geographic distance (Euclidean in lat/lon).

    Returns
    -------
    nbr_idx  : int64 ndarray (n, k)   — neighbour indices (excludes self)
    nbr_dist : float32 ndarray (n, k) — Euclidean distances
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree",
                          metric="euclidean", n_jobs=-1)
    nn.fit(coords)
    dist, idx = nn.kneighbors(coords)
    # idx[:,0] == self → drop it
    return idx[:, 1:].astype(np.int64), dist[:, 1:].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SpatialRegressionDataset(Dataset):
    """
    Each sample = (query_idx,) but __getitem__ returns everything the model
    needs for that query point.

    Parameters
    ----------
    coords   : (n, 2) float32
    X        : (n, p) float32
    y        : (n,)   float32
    nbr_idx  : (n, k) int64
    nbr_dist : (n, k) float32
    """

    def __init__(self, coords, X, y, nbr_idx, nbr_dist):
        self.coords   = torch.from_numpy(coords)
        self.X        = torch.from_numpy(X)
        self.y        = torch.from_numpy(y)
        self.nbr_idx  = torch.from_numpy(nbr_idx)
        self.nbr_dist = torch.from_numpy(nbr_dist)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        nbr = self.nbr_idx[idx]          # (k,)
        return {
            "idx":        idx,
            "coord":      self.coords[idx],        # (2,)
            "x":          self.X[idx],             # (p,)
            "y":          self.y[idx],             # scalar
            "nbr_coord":  self.coords[nbr],        # (k, 2)
            "nbr_x":      self.X[nbr],             # (k, p)
            "nbr_y":      self.y[nbr],             # (k,)
            "nbr_dist":   self.nbr_dist[idx],      # (k,)
        }


def get_dataloaders(source="synthetic", k=16, val_split=0.2,
                    batch_size=256, seed=42, n_samples=5000):
    """
    Build train/val DataLoaders.

    Parameters
    ----------
    source : "synthetic" | "california"
    """
    if source == "california":
        coords, X, y = load_california_housing()
    else:
        coords, X, y = make_synthetic_spatial(n=n_samples, seed=seed)

    nbr_idx, nbr_dist = build_knn_graph(coords, k)

    dataset = SpatialRegressionDataset(coords, X, y, nbr_idx, nbr_dist)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed))

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  drop_last=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, drop_last=False, num_workers=0)

    feat_dim = X.shape[1]
    return train_dl, val_dl, feat_dim
