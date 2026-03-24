"""
gwf/models/bridges.py
=====================
Module 2: MLP Bridges

Projects each encoder channel from its native dimension into the shared
d=512 space so CrossChannelFusion can compare and combine them.

Architecture per bridge:
    bridge(z) = LayerNorm( MLP(z) + skip(z) )
    MLP  = Linear(d_in, d) → GELU → LayerNorm(d) → Linear(d, d)
    skip = Linear(d_in, d)  if d_in ≠ d
         = Identity          if d_in == d

AllMLPBridges is initialised from FrozenEncoderBank.out_dims so that it
automatically handles whatever hidden_size the LLM checkpoint has.
"""

import torch
import torch.nn as nn


class MLPBridge(nn.Module):
    """
    Single residual MLP bridge: d_in → d.

    bridge(z) = LayerNorm( MLP(z) + skip(z) )
    """

    def __init__(self, d_in: int, d: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )
        self.skip = nn.Linear(d_in, d, bias=False) if d_in != d else nn.Identity()
        self.norm = nn.LayerNorm(d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, d_in) → (N, d)"""
        return self.norm(self.mlp(z) + self.skip(z))


class AllMLPBridges(nn.Module):
    """
    One MLPBridge per encoder channel.

    channel_dims : dict mapping channel name → native output dimension.
                   Obtained from FrozenEncoderBank.out_dims after construction.
    d            : shared embedding dimension (default 512).

    Input : z_dict  {channel: (N, d_channel)}
    Output: h_dict  {channel: (N, d)}  — all in shared space
    """

    # Canonical channel order (must match CrossChannelFusion.CHANNELS)
    CHANNELS = ["satclip", "geoclip", "skysense", "anygraph", "tabfpn", "llm"]

    def __init__(self, channel_dims: dict[str, int], d: int = 512):
        super().__init__()
        self.d = d
        self.bridges = nn.ModuleDict({
            name: MLPBridge(channel_dims[name], d)
            for name in self.CHANNELS
        })

    def forward(self, z_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.bridges[name](z_dict[name]) for name in self.CHANNELS}
