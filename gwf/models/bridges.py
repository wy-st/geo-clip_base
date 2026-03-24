"""
gwf/models/bridges.py
=====================
Module 2: MLP Bridges

Each encoder channel outputs a different dimension. MLP bridges project
all channels into the same d=512 dimensional space so the fusion layer
can compare and combine them meaningfully.

Architecture per bridge:
    bridge(z) = LayerNorm(MLP(z) + skip(z))
    MLP  = Linear(d_in, d) -> GELU -> LayerNorm(d) -> Linear(d, d)
    skip = Linear(d_in, d) if d_in != d, else Identity
"""

import torch
import torch.nn as nn


# =============================================================================
# Single MLP Bridge
# =============================================================================

class MLPBridge(nn.Module):
    """
    Projects one encoder channel from d_in → d with a residual MLP.

    bridge(z) = LayerNorm( MLP(z) + skip(z) )
      MLP  : Linear(d_in,d) → GELU → LayerNorm → Linear(d,d)
      skip : Linear(d_in,d) if d_in≠d, else Identity
    """

    def __init__(self, d_in: int, d: int = 512):
        super().__init__()

        # Main MLP path
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )

        # Skip (residual) connection
        self.skip = nn.Linear(d_in, d, bias=False) if d_in != d else nn.Identity()

        # Output normalisation
        self.norm = nn.LayerNorm(d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, d_in) → (N, d)"""
        return self.norm(self.mlp(z) + self.skip(z))


# =============================================================================
# AllMLPBridges — one bridge per encoder channel
# =============================================================================

class AllMLPBridges(nn.Module):
    """
    Creates one MLPBridge per encoder channel and applies them to a z_dict.

    Input  : z_dict — output of FrozenEncoderBank
        {
            "satclip":  (N, 512),
            "geoclip":  (N, 512),
            "skysense": (N, 768),
            "anygraph": (N, 256),
            "tabfpn":   (N, 512),
            "llm":      (N, 1024),
        }
    Output : h_dict — projected to shared space
        {
            "satclip":  (N, d),
            "geoclip":  (N, d),
            "skysense": (N, d),
            "anygraph": (N, d),
            "tabfpn":   (N, d),   ← used later in FiLM as x features
            "llm":      (N, d),   ← used as task context c_task
        }
    """

    # Native output dimensions of each frozen encoder
    CHANNEL_DIMS = {
        "satclip":  512,
        "geoclip":  512,
        "skysense": 768,
        "anygraph": 256,
        "tabfpn":   512,
        "llm":      1024,
    }

    def __init__(self, d: int = 512):
        super().__init__()
        self.d = d
        self.bridges = nn.ModuleDict({
            name: MLPBridge(d_in, d)
            for name, d_in in self.CHANNEL_DIMS.items()
        })

    def forward(self, z_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply each bridge to its corresponding frozen embedding."""
        return {
            name: self.bridges[name](z)
            for name, z in z_dict.items()
        }
