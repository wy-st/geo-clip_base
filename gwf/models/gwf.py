"""
gwf/models/gwf.py
=================
GWF: Geographically Weighted Foundation Model

Main model class that assembles all 8 modules into the complete forward pass.

                     ┌──────────────────────────────────────────┐
  coords, X_tab,     │  FrozenEncoderBank (no grad)             │
  images?, graphs?,  │  SatCLIP · GeoCLIP · SkySense · AnyGraph │
  prompt_text?  ────▶│  TabPFN  · LLM                           │──▶ z_dict
                     └──────────────────────────────────────────┘
                                                                        │
                     ┌──────────────────────────────────────────┐       │
                     │  AllMLPBridges  (trainable)               │◀──────┘
                     │  6 × MLPBridge(d_in → 512)               │──▶ h_dict
                     └──────────────────────────────────────────┘
                                                                        │
                     ┌──────────────────────────────────────────┐       │
                     │  CrossChannelFusion  (trainable)          │◀──────┘
                     │  Gated attention over 6 channels          │──▶ h_fused, attn_w
                     └──────────────────────────────────────────┘
                                                                        │
                     ┌──────────────────────────────────────────┐       │
                     │  SpatialGraphBuilder  (trainable)         │◀──────┘
                     │  kNN graph + learned kernel               │──▶ edge_index, edge_w
                     └──────────────────────────────────────────┘
                                                                        │
                     ┌──────────────────────────────────────────┐       │
                     │  SpatialGNN  (trainable)                  │◀──────┘
                     │  L=3 GeoWeightedConv layers               │──▶ h_spatial
                     └──────────────────────────────────────────┘
                                                                        │
                     ┌──────────────────────────────────────────┐       │
                     │  HyperNetBeta  (trainable)                │◀──────┘
                     │  Generates Beta = {U, sigma, V}          │──▶ Beta
                     └──────────────────────────────────────────┘
                                                                        │
                     ┌──────────────────────────────────────────┐       │
                     │  FiLMLayer  (trainable)                   │◀──────┘
                     │  Beta modulates h_tabfpn                  │──▶ h_pred
                     └──────────────────────────────────────────┘
                                                                        │
                     ┌──────────────────────────────────────────┐       │
                     │  OutputHead  (trainable)                  │◀──────┘
                     │  Linear(d, d/2) → GELU → Linear(d/2, T) │──▶ y_pred
                     └──────────────────────────────────────────┘
"""

import torch
import torch.nn as nn

from gwf.models.encoders     import FrozenEncoderBank
from gwf.models.bridges      import AllMLPBridges
from gwf.models.fusion       import CrossChannelFusion
from gwf.models.graph_builder import SpatialGraphBuilder
from gwf.models.gnn          import SpatialGNN
from gwf.models.hypernet     import HyperNetBeta
from gwf.models.film         import FiLMLayer
from gwf.models.heads        import OutputHead


class GWF(nn.Module):
    """
    Geographically Weighted Foundation Model.

    All foundation models (FrozenEncoderBank) are frozen — gradients only
    flow through the 7 trainable modules:
        AllMLPBridges, CrossChannelFusion, SpatialGraphBuilder,
        SpatialGNN, HyperNetBeta, FiLMLayer, OutputHead

    Args:
        cfg          : dict with all hyperparameters (see config.py)
        num_targets  : 1 for regression, num_classes for classification
        probabilistic: False → V1 deterministic, True → V2 variational
    """

    def __init__(
        self,
        cfg:           dict,
        num_targets:   int  = 1,
        probabilistic: bool = False,
    ):
        super().__init__()

        d   = cfg.get("d",              512)
        r   = cfg.get("r",               32)
        k   = cfg.get("k_neighbors",     15)
        L   = cfg.get("num_gnn_layers",   3)

        # ---- Module 1: Frozen Encoder Bank (always frozen) ----
        self.encoder_bank = FrozenEncoderBank(cfg)

        # ---- Module 2: MLP Bridges (trainable) ----
        self.bridges = AllMLPBridges(d=d)

        # ---- Module 3: Cross-Channel Fusion (trainable) ----
        self.fusion = CrossChannelFusion(d=d)

        # ---- Module 4: Spatial Graph Builder (trainable) ----
        self.graph_builder = SpatialGraphBuilder(d=d, k=k)

        # ---- Module 5: Spatial GNN (trainable) ----
        self.gnn = SpatialGNN(d=d, num_layers=L)

        # ---- Module 6: HyperNet (trainable) ----
        self.hypernet = HyperNetBeta(d=d, r=r, probabilistic=probabilistic)

        # ---- Module 7: FiLM Prediction Layer (trainable) ----
        self.film = FiLMLayer(d=d)

        # ---- Module 8: Output Head (trainable) ----
        self.head = OutputHead(d=d, num_targets=num_targets)

    # -------------------------------------------------------------------------
    # Convenience: count trainable parameters
    # -------------------------------------------------------------------------

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------

    def forward(
        self,
        coords:       torch.Tensor,          # (N, 2)  lat/lon  REQUIRED
        X_tab:        torch.Tensor,          # (N, p)  tabular  REQUIRED
        y:            torch.Tensor | None = None,  # (N,) labels for TabPFN context
        images:       torch.Tensor | None = None,  # (N, C, H, W) optional
        subgraphs=None,                      # list[PyG Data] optional
        prompt_text:  str           = "",    # task description for LLM
        deterministic: bool         = False, # V2 only: use posterior means
    ) -> tuple[torch.Tensor, dict]:
        """
        Complete forward pass.

        Returns:
            y_pred   (N, num_targets)
            aux_dict  — contains all intermediate tensors needed for loss / analysis:
                "h_dict"     : bridged channel embeddings
                "h_fused"    : unified point embedding  (N, d)
                "attn_weights": modality importance     (N, 6)
                "edge_index" : (2, E)
                "edge_weight": (E,)
                "h_spatial"  : spatially contextualised (N, d)
                "Beta"       : {"U","sigma","V"[,"kl_loss"]}
                "h_pred"     : pre-output embedding    (N, d)
        """
        # ---- Module 1: Frozen encoders ----
        with torch.no_grad():
            z_dict = self.encoder_bank(
                coords=coords,
                X_tab=X_tab,
                y=y,
                images=images,
                subgraphs=subgraphs,
                prompt_text=prompt_text,
            )

        # ---- Module 2: MLP bridges ----
        h_dict = self.bridges(z_dict)          # {channel: (N, d)}

        # ---- Module 3: Cross-channel fusion ----
        h_fused, attn_weights = self.fusion(h_dict)  # (N, d), (N, 6)

        # ---- Module 4: Spatial graph ----
        c_task = h_dict["llm"]                 # (N, d)  task context
        edge_index, edge_weight = self.graph_builder(h_fused, coords, c_task)

        # ---- Module 5: Spatial GNN ----
        h_spatial = self.gnn(h_fused, edge_index, edge_weight)  # (N, d)

        # ---- Module 6: HyperNet → Beta ----
        beta = self.hypernet(h_spatial, c_task, deterministic=deterministic)

        # ---- Module 7: FiLM prediction ----
        h_tabfpn = h_dict["tabfpn"]            # (N, d)  tabular features
        h_pred   = self.film(h_tabfpn, beta)   # (N, d)

        # ---- Module 8: Output head ----
        y_pred = self.head(h_pred)             # (N, num_targets)

        aux_dict = {
            "h_dict":      h_dict,
            "h_fused":     h_fused,
            "attn_weights": attn_weights,
            "edge_index":  edge_index,
            "edge_weight": edge_weight,
            "h_spatial":   h_spatial,
            "Beta":        beta,
            "h_pred":      h_pred,
        }

        return y_pred, aux_dict

    # -------------------------------------------------------------------------
    # Phase-specific parameter freeze helpers (used by train.py)
    # -------------------------------------------------------------------------

    def freeze_for_phase1(self):
        """Phase 1: train ONLY bridges + fusion."""
        # Freeze everything first
        for p in self.parameters():
            p.requires_grad_(False)
        # Unfreeze bridges and fusion
        for p in self.bridges.parameters():
            p.requires_grad_(True)
        for p in self.fusion.parameters():
            p.requires_grad_(True)

    def freeze_for_phase2(self):
        """Phase 2: train ALL trainable modules (unfreeze everything except encoder_bank)."""
        # encoder_bank is always frozen (set in FrozenEncoderBank itself)
        for p in self.parameters():
            p.requires_grad_(True)
        # Re-freeze encoder_bank
        for p in self.encoder_bank.parameters():
            p.requires_grad_(False)

    def freeze_for_phase3(self):
        """Phase 3: train ONLY output_head + FiLM + MLP_out (film.mlp_out)."""
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.film.parameters():
            p.requires_grad_(True)
        for p in self.head.parameters():
            p.requires_grad_(True)
