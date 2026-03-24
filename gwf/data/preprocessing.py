"""
gwf/data/preprocessing.py
==========================
Feature preprocessing utilities for GWF.

Includes:
  - StandardScaler: z-normalise feature columns
  - TargetTransformer: log / identity transform on y
  - extract_local_subgraph: build POI/road subgraph from OSM data
    (requires osmnx + torch_geometric; skipped if not available)
"""

import numpy as np
import torch
from typing import Optional


# =============================================================================
# StandardScaler (torch-compatible)
# =============================================================================

class StandardScaler:
    """
    Computes per-feature mean and std from training data, then scales
    both training and validation tensors consistently.

    Usage:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
    """

    def __init__(self, eps: float = 1e-8):
        self.eps  = eps
        self.mean: Optional[torch.Tensor] = None
        self.std:  Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "StandardScaler":
        self.mean = X.mean(dim=0)
        self.std  = X.std(dim=0).clamp(min=self.eps)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None, "Call fit() first."
        return (X - self.mean) / self.std

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None, "Call fit() first."
        return X_scaled * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict):
        self.mean = state["mean"]
        self.std  = state["std"]


# =============================================================================
# Target Transformer
# =============================================================================

class TargetTransformer:
    """
    Applies a monotonic transform to the regression target y.

    Transforms:
        "log"      : y' = log(y + offset)    (default offset=1)
        "log1p"    : y' = log1p(y)
        "identity" : y' = y
        "sqrt"     : y' = sqrt(y)

    For "log", y must be positive.

    Usage:
        tt = TargetTransformer("log1p")
        y_train_t = tt.transform(y_train)
        y_pred_orig = tt.inverse_transform(y_pred_t)
    """

    def __init__(self, mode: str = "identity", offset: float = 1.0):
        assert mode in ("log", "log1p", "identity", "sqrt"), \
            f"Unknown mode: {mode!r}"
        self.mode   = mode
        self.offset = offset

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "log":
            return torch.log(y + self.offset)
        elif self.mode == "log1p":
            return torch.log1p(y)
        elif self.mode == "sqrt":
            return torch.sqrt(y.clamp(min=0))
        else:  # identity
            return y

    def inverse_transform(self, y_t: torch.Tensor) -> torch.Tensor:
        if self.mode == "log":
            return torch.exp(y_t) - self.offset
        elif self.mode == "log1p":
            return torch.expm1(y_t)
        elif self.mode == "sqrt":
            return y_t.pow(2)
        else:
            return y_t


# =============================================================================
# Local Subgraph Extraction  [OPTIONAL — requires osmnx + torch_geometric]
# =============================================================================

def extract_local_subgraph(lat: float, lon: float, radius_m: float = 1000):
    """
    Extract a local POI/road subgraph centred at (lat, lon) using OSMnx.

    Returns a torch_geometric.data.Data object with:
        x         : (num_nodes, node_feat_dim) node features
        edge_index: (2, num_edges)
        edge_attr : (num_edges, edge_feat_dim) edge features (length, speed)

    If osmnx or torch_geometric is not installed, returns None with a warning.

    Args:
        lat      : latitude of centre point
        lon      : longitude of centre point
        radius_m : radius in metres for graph extraction (default 1km)
    """
    try:
        import osmnx as ox                                # type: ignore
        from torch_geometric.data import Data             # type: ignore
    except ImportError as e:
        import warnings
        warnings.warn(
            f"Cannot extract subgraph: {e}. "
            "Install osmnx and torch_geometric for graph channel support."
        )
        return None

    # Download road network within radius
    G = ox.graph_from_point((lat, lon), dist=radius_m, network_type="all")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    nodes, edges = ox.graph_to_gdfs(G)

    # Node features: [x_coord, y_coord] (normalised)
    node_x = torch.tensor(nodes[["x", "y"]].values, dtype=torch.float32)
    node_x = (node_x - node_x.mean(0)) / node_x.std(0).clamp(1e-8)

    # Build edge_index
    node_ids    = list(G.nodes)
    node_id_map = {nid: i for i, nid in enumerate(node_ids)}
    edge_list   = []
    edge_feats  = []
    for u, v, data in G.edges(data=True):
        if u in node_id_map and v in node_id_map:
            edge_list.append([node_id_map[u], node_id_map[v]])
            length    = data.get("length",      0.0)
            speed     = data.get("speed_kph",   30.0)
            trav_time = data.get("travel_time", 0.0)
            edge_feats.append([length, speed, trav_time])

    if len(edge_list) == 0:
        return None

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_feats, dtype=torch.float32)
    # Normalise edge features
    edge_attr  = (edge_attr - edge_attr.mean(0)) / edge_attr.std(0).clamp(1e-8)

    return Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr)


def extract_subgraphs_batch(
    coords:   np.ndarray,
    radius_m: float = 1000,
    verbose:  bool  = True,
) -> list:
    """
    Extract local subgraphs for a batch of coordinates.

    Args:
        coords   : (N, 2) numpy array of (lat, lon)
        radius_m : extraction radius in metres
        verbose  : print progress

    Returns:
        list of PyG Data objects (or None where extraction failed)
    """
    subgraphs = []
    N = coords.shape[0]
    for i in range(N):
        if verbose and i % 100 == 0:
            print(f"  Extracting subgraphs: {i}/{N}", end="\r")
        sg = extract_local_subgraph(coords[i, 0], coords[i, 1], radius_m)
        subgraphs.append(sg)
    if verbose:
        n_ok = sum(sg is not None for sg in subgraphs)
        print(f"\n  Subgraph extraction: {n_ok}/{N} successful.")
    return subgraphs
