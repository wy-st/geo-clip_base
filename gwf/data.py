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
# Custom dataset loader
# ──────────────────────────────────────────────────────────────────────────────

def load_custom_dataset(path: str,
                        lat_col:    str = "lat",
                        lon_col:    str = "lon",
                        target_col: str = "price",
                        feature_cols: list | None = None,
                        standardize_y: bool = True):
    """
    Load your own CSV dataset into the format expected by GWF.

    Parameters
    ----------
    path          : path to the CSV file
    lat_col       : name of the latitude column  (decimal degrees)
    lon_col       : name of the longitude column (decimal degrees)
    target_col    : name of the column to predict
    feature_cols  : list of feature column names to use as predictors.
                    If None, uses ALL columns except lat, lon, and target.
    standardize_y : whether to z-score normalise the target (recommended)

    Returns
    -------
    coords : float32 ndarray (n, 2)  — [lat, lon]
    X      : float32 ndarray (n, p)  — standardised tabular features
    y      : float32 ndarray (n,)    — (optionally standardised) target

    Example
    -------
    coords, X, y = load_custom_dataset(
        "housing.csv",
        lat_col="latitude",
        lon_col="longitude",
        target_col="price",
        feature_cols=["area", "rooms", "age", "dist_subway"],
    )
    """
    import pandas as pd

    df = pd.read_csv(path)

    # ── Coordinates ───────────────────────────────────────────────────────
    coords = df[[lat_col, lon_col]].values.astype(np.float32)

    # ── Feature columns ───────────────────────────────────────────────────
    if feature_cols is None:
        exclude = {lat_col, lon_col, target_col}
        feature_cols = [c for c in df.columns if c not in exclude]

    X_raw = df[feature_cols].values.astype(np.float64)

    # Handle missing values: fill with column median
    col_medians = np.nanmedian(X_raw, axis=0)
    nan_mask = np.isnan(X_raw)
    X_raw[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    # ── Target ────────────────────────────────────────────────────────────
    y = df[target_col].values.astype(np.float32)
    if standardize_y:
        y = ((y - y.mean()) / (y.std() + 1e-8))

    print(f"[load_custom_dataset] n={len(y)}, p={X.shape[1]}, "
          f"features={feature_cols}")
    print(f"  lat ∈ [{coords[:,0].min():.3f}, {coords[:,0].max():.3f}]  "
          f"lon ∈ [{coords[:,1].min():.3f}, {coords[:,1].max():.3f}]")

    return coords, X, y.astype(np.float32)


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

        # 预计算的 TabPFN 嵌入（可选）
        # 设置后 __getitem__ 会把它们加入 batch，model.forward 会直接用
        self.z_query: torch.Tensor | None = None  # (n, D)
        self.z_nbr:   torch.Tensor | None = None  # (n, k, D)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        nbr  = self.nbr_idx[idx]          # (k,)
        item = {
            "idx":        idx,
            "coord":      self.coords[idx],        # (2,)
            "x":          self.X[idx],             # (p,)
            "y":          self.y[idx],             # scalar
            "nbr_coord":  self.coords[nbr],        # (k, 2)
            "nbr_x":      self.X[nbr],             # (k, p)
            "nbr_y":      self.y[nbr],             # (k,)
            "nbr_dist":   self.nbr_dist[idx],      # (k,)
        }
        if self.z_query is not None:
            item["z_query"] = self.z_query[idx]    # (D,)
            item["z_nbr"]   = self.z_nbr[idx]      # (k, D)
        return item


def load_geojson_dataset(path: str,
                         target_col: str = "log_price",
                         feature_cols: list | None = None,
                         standardize_y: bool = True):
    """
    Load a GeoJSON FeatureCollection (Point geometry) into the GWF format.

    Coordinates are read from geometry.coordinates = [lon, lat].

    Returns
    -------
    coords : float32 ndarray (n, 2)  — [lat, lon]
    X      : float32 ndarray (n, p)  — standardised tabular features
    y      : float32 ndarray (n,)    — (optionally standardised) target
    """
    import json
    import pandas as pd

    with open(path) as f:
        geojson = json.load(f)

    features = geojson["features"]
    lats, lons, targets = [], [], []
    prop_rows = []
    for feat in features:
        lon, lat = feat["geometry"]["coordinates"]
        lats.append(lat); lons.append(lon)
        prop_rows.append(feat["properties"])
        targets.append(feat["properties"][target_col])

    props_df = pd.DataFrame(prop_rows)

    if feature_cols is None:
        exclude = {target_col}
        numeric_cols = props_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude]

    coords = np.stack([lats, lons], axis=1).astype(np.float32)
    X_raw  = props_df[feature_cols].values.astype(np.float64)

    col_medians = np.nanmedian(X_raw, axis=0)
    nan_mask = np.isnan(X_raw)
    X_raw[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    y = np.array(targets, dtype=np.float32)
    if standardize_y:
        y = ((y - y.mean()) / (y.std() + 1e-8)).astype(np.float32)

    print(f"[load_geojson] n={len(y)}, p={X.shape[1]}, features={feature_cols}")
    print(f"  lat ∈ [{coords[:,0].min():.3f}, {coords[:,0].max():.3f}]  "
          f"lon ∈ [{coords[:,1].min():.3f}, {coords[:,1].max():.3f}]")

    return coords, X, y


def get_dataloaders(source="synthetic", k=16, val_split=0.2,
                    batch_size=256, seed=42, n_samples=5000,
                    # ── custom / geojson dataset options ───────────────────
                    csv_path:     str | None = None,
                    geojson_path: str | None = None,
                    lat_col:    str = "lat",
                    lon_col:    str = "lon",
                    target_col: str = "price",
                    feature_cols: list | None = None,
                    coords: np.ndarray | None = None,
                    X:      np.ndarray | None = None,
                    y:      np.ndarray | None = None):
    """
    Build train/val DataLoaders.

    Parameters
    ----------
    source       : "synthetic" | "california" | "custom"
                   Use "custom" when providing csv_path or arrays directly.
    csv_path     : path to your CSV file (when source="custom")
    lat_col      : latitude column name in your CSV
    lon_col      : longitude column name in your CSV
    target_col   : target column name in your CSV
    feature_cols : list of feature column names; None = all except lat/lon/target
    coords       : (n,2) float32 ndarray — pass arrays directly instead of CSV
    X            : (n,p) float32 ndarray
    y            : (n,)  float32 ndarray
    """
    if source == "california":
        coords, X, y = load_california_housing()
    elif source == "geojson":
        if geojson_path is None:
            raise ValueError("source='geojson' requires geojson_path=")
        coords, X, y = load_geojson_dataset(
            geojson_path, target_col=target_col, feature_cols=feature_cols)
    elif source == "custom":
        if coords is not None and X is not None and y is not None:
            # Arrays passed directly — use as-is
            pass
        elif csv_path is not None:
            coords, X, y = load_custom_dataset(
                csv_path, lat_col=lat_col, lon_col=lon_col,
                target_col=target_col, feature_cols=feature_cols)
        else:
            raise ValueError(
                "source='custom' requires either csv_path= or (coords, X, y) arrays.")
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
