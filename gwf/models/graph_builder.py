"""
gwf/models/graph_builder.py
============================
Module 4: Spatial Graph Builder

Builds a k-nearest-neighbour graph (based on geographic coords) and computes
learned edge weights w_ij ∈ [0, 1] that encode both geographic AND semantic
distance decay — the GWF equivalent of GWR's kernel function.

Two backends for kNN construction:
  1. torch_geometric.nn.knn_graph   (fast, preferred if PyG is installed)
  2. scipy.spatial.cKDTree fallback (always available, slightly slower)

Edge weight input per edge (i, j):
  d_geo: normalised geographic distance  scalar
  d_sem: normalised semantic distance    scalar
  c_ctx: task-conditioned context       (d,)
  → kernel_input: (d+2,) → MLP → w_ij scalar ∈ [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _knn_graph_scipy(coords: torch.Tensor, k: int) -> torch.Tensor:
    """
    Fallback kNN graph builder using scipy.spatial.cKDTree.
    Returns edge_index (2, N*k) in COO format (row=target, col=source).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    xy = coords.detach().cpu().numpy()
    tree = cKDTree(xy)
    # Query k+1 because the point itself is the closest neighbor
    dists, idx = tree.query(xy, k=k + 1)
    # idx[:, 0] is the point itself; skip it
    neighbors = idx[:, 1:]          # (N, k)

    N = xy.shape[0]
    sources = neighbors.reshape(-1)                     # (N*k,)
    targets = np.repeat(np.arange(N), k)                # (N*k,)
    edge_index = torch.tensor(
        np.stack([targets, sources], axis=0), dtype=torch.long
    )
    return edge_index.to(coords.device)


def _knn_graph_pyg(coords: torch.Tensor, k: int) -> torch.Tensor:
    """
    Fast kNN graph using torch_geometric.nn.knn_graph.
    Returns edge_index (2, N*k).
    """
    from torch_geometric.nn import knn_graph  # type: ignore
    return knn_graph(coords, k=k, loop=False)


def build_knn_graph(coords: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build kNN graph on geographic coordinates.
    Tries PyG first, falls back to scipy.
    Returns edge_index (2, N*k).
    """
    try:
        return _knn_graph_pyg(coords, k)
    except ImportError:
        return _knn_graph_scipy(coords, k)


# =============================================================================
# SpatialGraphBuilder
# =============================================================================

class SpatialGraphBuilder(nn.Module):
    """
    Constructs a spatial kNN graph and computes learned edge weights.

    Step 1: Build kNN graph from geographic coordinates (hard neighbourhood).
    Step 2: Compute learned edge weight w_ij for every edge:
              w_ij = sigmoid(MLP_kernel([d_geo, d_sem, c_ctx]))

    The MLP kernel learns geographic + semantic distance decay, conditioned
    on the task context (LLM embedding mean over the batch).

    Args:
        d   : embedding dimension (default 512)
        k   : number of nearest neighbours (default 15)
    """

    def __init__(self, d: int = 512, k: int = 15):
        super().__init__()
        self.d = d
        self.k = k

        # Task context projection: c_task_mean (d,) → (d,)
        self.ctx_proj = nn.Linear(d, d, bias=False)

        # Kernel MLP: [d_geo(1), d_sem(1), c_ctx(d)] → w_ij(1) ∈ [0,1]
        self.kernel_mlp = nn.Sequential(
            nn.Linear(d + 2, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_fused: torch.Tensor,          # (N, d) unified point embeddings
        coords:  torch.Tensor,          # (N, 2) lat/lon
        c_task:  torch.Tensor,          # (N, d) task context per point
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            edge_index  (2, E)  E = N*k
            edge_weight (E,)    ∈ [0, 1]
        """
        N = coords.shape[0]
        device = coords.device

        # ---- Step 1: kNN graph ------------------------------------------------
        edge_index = build_knn_graph(coords, self.k)   # (2, E)
        src = edge_index[1]   # source nodes (j)
        dst = edge_index[0]   # destination nodes (i)

        # ---- Step 2: Normalised geographic distance ---------------------------
        coords_src = coords[src]     # (E, 2)
        coords_dst = coords[dst]     # (E, 2)
        geo_dist = (coords_src - coords_dst).norm(dim=-1, keepdim=True)  # (E, 1)
        max_geo  = geo_dist.max().clamp(min=1e-8)
        d_geo = geo_dist / max_geo                                        # (E, 1)

        # ---- Step 3: Normalised semantic distance -----------------------------
        h_src = h_fused[src]         # (E, d)
        h_dst = h_fused[dst]         # (E, d)
        sem_dist = (h_src - h_dst).norm(dim=-1, keepdim=True)            # (E, 1)
        max_sem  = sem_dist.max().clamp(min=1e-8)
        d_sem = sem_dist / max_sem                                        # (E, 1)

        # ---- Step 4: Task context (mean across batch, projected) --------------
        c_mean = c_task.mean(dim=0, keepdim=True)                        # (1, d)
        c_ctx  = self.ctx_proj(c_mean).expand(edge_index.shape[1], -1)  # (E, d)

        # ---- Step 5: Kernel MLP → edge weight --------------------------------
        kernel_input = torch.cat([d_geo, d_sem, c_ctx], dim=-1)         # (E, d+2)
        edge_weight  = self.kernel_mlp(kernel_input).squeeze(-1)        # (E,)

        return edge_index, edge_weight
