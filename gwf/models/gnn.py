"""
gwf/models/gnn.py
=================
Module 5: Spatial GNN

GeoWeightedConv: one message-passing layer where edge weights (from the
SpatialGraphBuilder) multiply neighbor messages, implementing geographic +
semantic distance decay.

SpatialGNN: stacks L=3 GeoWeightedConv layers. After L layers each point
has aggregated information from its L-hop spatial neighbourhood.

Architecture per layer:
  Message  : msg_ji = w_ji * MLP_msg(h_j)
  Aggregate: agg_i  = Σ_{j ∈ N(i)} msg_ji
  Update   : h_i'   = LayerNorm(MLP_update([h_i, agg_i]) + h_i)

Two backends:
  1. torch_geometric.nn.MessagePassing (preferred if PyG available)
  2. Pure-PyTorch scatter-add fallback (always works, slightly slower)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Fallback: scatter-add without PyG
# =============================================================================

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter-add: out[index[i]] += src[i]"""
    out = torch.zeros(dim_size, src.shape[-1], dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    return out


# =============================================================================
# GeoWeightedConv — single message-passing layer
# =============================================================================

class GeoWeightedConv(nn.Module):
    """
    One round of weighted spatial message passing.

    For each edge (j → i) with weight w_ij:
        msg_ji = w_ij * MLP_msg(h_j)
    Aggregation:
        agg_i  = Σ msg_ji
    Update (with residual):
        h_i'   = LayerNorm(MLP_update([h_i, agg_i]) + h_i)

    Args:
        d : embedding dimension
    """

    def __init__(self, d: int = 512):
        super().__init__()
        self.d = d

        # Message MLP: transforms source node features
        self.mlp_msg = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Update MLP: combines self + aggregated neighbourhood
        self.mlp_update = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        h:           torch.Tensor,   # (N, d)
        edge_index:  torch.Tensor,   # (2, E)  [dst, src]
        edge_weight: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        """Returns updated node features (N, d)."""
        N = h.shape[0]
        dst = edge_index[0]   # target/destination nodes
        src = edge_index[1]   # source nodes

        # --- Message ---
        msg_raw = self.mlp_msg(h[src])              # (E, d)
        msg     = edge_weight.unsqueeze(-1) * msg_raw  # (E, d)  weighted

        # --- Aggregate ---
        agg = scatter_add(msg, dst, dim_size=N)     # (N, d)

        # --- Update (residual) ---
        combined = torch.cat([h, agg], dim=-1)      # (N, 2d)
        h_new    = self.norm(self.mlp_update(combined) + h)  # (N, d)

        return h_new


# =============================================================================
# SpatialGNN — stacks L GeoWeightedConv layers
# =============================================================================

class SpatialGNN(nn.Module):
    """
    Stacks L=3 GeoWeightedConv layers.

    Each layer expands the receptive field by one hop. After L layers each
    point has aggregated information from roughly its K^L spatial neighbours
    (though the kNN graph only has 1-hop edges; information propagates via
    consecutive layers).

    Input : h_fused (N, d),  edge_index (2, E),  edge_weight (E,)
    Output: h_spatial (N, d)  — spatially contextualised embeddings
    """

    def __init__(self, d: int = 512, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [GeoWeightedConv(d) for _ in range(num_layers)]
        )

    def forward(
        self,
        h:           torch.Tensor,   # (N, d)
        edge_index:  torch.Tensor,   # (2, E)
        edge_weight: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        """Returns spatially contextualised node embeddings (N, d)."""
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        return h
