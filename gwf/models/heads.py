"""
gwf/models/heads.py
===================
Module 8: Output Head

Maps the d-dimensional prediction representation (from FiLMLayer) to the
final target output.

Architecture:
    Linear(d, d//2) → GELU → Linear(d//2, num_targets)

Regression   : num_targets=1, outputs raw scalar
Classification: num_targets=num_classes, apply softmax in loss function
"""

import torch
import torch.nn as nn


class OutputHead(nn.Module):
    """
    Two-layer output MLP.

    Args:
        d           : input embedding dimension (default 512)
        num_targets : output dimension.  1 for regression, >1 for classification
    """

    def __init__(self, d: int = 512, num_targets: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, num_targets),
        )

    def forward(self, h_pred: torch.Tensor) -> torch.Tensor:
        """h_pred (N, d) → y_pred (N, num_targets)"""
        return self.net(h_pred)
