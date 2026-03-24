"""
gwf/losses/task_loss.py
========================
Task Loss: MSE for regression, CrossEntropy for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskLoss(nn.Module):
    """
    Regression : MSE(y_pred, y_true)
    Classification: CrossEntropy(y_pred, y_true)

    Args:
        task : "regression" or "classification"
    """

    def __init__(self, task: str = "regression"):
        super().__init__()
        assert task in ("regression", "classification"), \
            f"task must be 'regression' or 'classification', got '{task}'"
        self.task = task

    def forward(
        self,
        y_pred: torch.Tensor,   # (N, 1) or (N, num_classes)
        y_true: torch.Tensor,   # (N,)
    ) -> torch.Tensor:
        if self.task == "regression":
            return F.mse_loss(y_pred.squeeze(-1), y_true.float())
        else:
            return F.cross_entropy(y_pred, y_true.long())
