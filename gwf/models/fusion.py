"""
gwf/models/fusion.py
====================
Module 3: Cross-Channel Fusion

Combines all 6 bridged channel embeddings into a single unified point
embedding h_fused using gated attention.

Why gated attention instead of concatenation or averaging?
  Different modalities are important in different geographic contexts.
  Urban areas: POI graph + street-view matter most.
  Rural areas: satellite imagery + terrain features matter most.
  The gated attention mechanism LEARNS these location-dependent weights.

Architecture:
  Step 1: Stack 6 channel embeddings → (N, 6, d)
  Step 2: Attention weights per channel per point via small MLP → softmax
  Step 3: Weighted sum → h_attn (N, d)
  Step 4: Gate from full concatenation → sigmoid → (N, d)
  Step 5: h_fused = LayerNorm(Linear(d,d)(gate * h_attn))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossChannelFusion(nn.Module):
    """
    Gated cross-channel fusion for 6 modality embeddings.

    Input : h_dict — dict of 6 tensors each (N, d)
    Output: h_fused (N, d)
            attn_weights (N, 6)  — per-channel attention weights (for analysis)
    """

    CHANNELS = ["satclip", "geoclip", "skysense", "anygraph", "tabfpn", "llm"]
    NUM_CHANNELS = 6

    def __init__(self, d: int = 512):
        super().__init__()
        self.d = d
        C = self.NUM_CHANNELS

        # Step 2: attention score MLP — d → d//4 → 1 (applied per token)
        self.attn_mlp = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
        )

        # Step 4: gate from concatenated channels — 6d → d
        self.gate_linear = nn.Linear(C * d, d)

        # Step 5: final projection + norm
        self.out_linear = nn.Linear(d, d)
        self.out_norm   = nn.LayerNorm(d)

    def forward(
        self,
        h_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h_dict : {channel_name: (N, d)} for each of 6 channels
        Returns:
            h_fused      (N, d)
            attn_weights (N, 6)  — interpretable modality weights
        """
        # --- Step 1: Stack all channels → (N, 6, d) ---
        # Use the fixed order defined in CHANNELS so output is deterministic
        channel_list = [h_dict[c] for c in self.CHANNELS]   # list of (N, d)
        stacked = torch.stack(channel_list, dim=1)            # (N, 6, d)

        # --- Step 2: Attention weights ---
        # Apply attn_mlp independently to each channel token
        logits = self.attn_mlp(stacked).squeeze(-1)           # (N, 6)
        attn_weights = F.softmax(logits, dim=1)               # (N, 6) sums to 1

        # --- Step 3: Weighted sum ---
        h_attn = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (N, d)

        # --- Step 4: Gate from full concatenation ---
        h_concat = torch.cat(channel_list, dim=-1)            # (N, 6d)
        gate = torch.sigmoid(self.gate_linear(h_concat))      # (N, d) ∈ [0,1]

        # --- Step 5: Gated output ---
        h_gated = gate * h_attn                               # (N, d)
        h_fused = self.out_norm(self.out_linear(h_gated))     # (N, d)

        return h_fused, attn_weights
