"""
gwf/losses/gwf_loss.py
=======================
GWFLoss: Combined loss function for all three training phases.

Phase 1  (bridge warm-up):
    L = L_task + lambda_contrast · L_contrast

Phase 2  (joint training):
    V1: L = L_task + lambda_smooth · L_smooth
    V2: L = L_task + lambda_smooth · L_smooth + L_KL (with annealing)

Phase 3  (task fine-tuning):
    L = L_task only

All λ weights are configured in config.py.
"""

import torch
import torch.nn as nn

from gwf.losses.task_loss        import TaskLoss
from gwf.losses.smooth_loss      import SpatialSmoothnessLoss
from gwf.losses.kl_loss          import KLLoss
from gwf.losses.contrastive_loss import GeographicContrastiveLoss


class GWFLoss(nn.Module):
    """
    Unified loss module for all GWF training phases.

    Args:
        cfg  : loss config dict (from config.LOSS)
        task : "regression" or "classification"
        probabilistic : whether V2 KL loss is active
    """

    def __init__(
        self,
        cfg:           dict,
        task:          str  = "regression",
        probabilistic: bool = False,
    ):
        super().__init__()
        self.probabilistic = probabilistic

        self.lambda_smooth   = cfg.get("lambda_smooth",   0.1)
        self.lambda_contrast = cfg.get("lambda_contrast", 0.1)

        self.task_loss    = TaskLoss(task=task)
        self.smooth_loss  = SpatialSmoothnessLoss(
            sigma_weight=cfg.get("smooth_sigma_weight", 1.0),
            u_weight=cfg.get("smooth_u_weight", 0.1),
        )
        self.kl_loss      = KLLoss(
            lambda_kl=cfg.get("lambda_kl", 0.01),
            kl_annealing_epochs=cfg.get("kl_annealing_epochs", 20),
        ) if probabilistic else None
        self.contrast_loss = GeographicContrastiveLoss(
            temperature=cfg.get("contrastive_temperature", 0.1),
        )

    def forward(
        self,
        y_pred:      torch.Tensor,          # (N, T)
        y_true:      torch.Tensor,          # (N,)
        aux_dict:    dict,                  # from GWF.forward()
        phase:       int   = 2,            # 1, 2, or 3
        epoch:       int   = 0,            # current epoch (for KL annealing)
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns (total_loss, loss_components).
        loss_components is a dict with float values for logging.
        """
        edge_index  = aux_dict["edge_index"]
        edge_weight = aux_dict["edge_weight"]
        beta        = aux_dict["Beta"]
        h_fused     = aux_dict["h_fused"]

        components: dict[str, float] = {}

        # ---- Task loss (all phases) ----
        l_task = self.task_loss(y_pred, y_true)
        components["task"] = l_task.item()

        if phase == 1:
            # ---- Phase 1: task + contrastive ----
            l_contrast = self.contrast_loss(h_fused, edge_index, edge_weight)
            components["contrast"] = l_contrast.item()
            total = l_task + self.lambda_contrast * l_contrast

        elif phase == 2:
            # ---- Phase 2: task + smooth [+ KL] ----
            l_smooth = self.smooth_loss(beta, edge_index, edge_weight)
            components["smooth"] = l_smooth.item()
            total = l_task + self.lambda_smooth * l_smooth

            if self.probabilistic and self.kl_loss is not None:
                l_kl = self.kl_loss(beta, epoch=epoch)
                components["kl"] = l_kl.item()
                total = total + l_kl

        else:
            # ---- Phase 3: task only ----
            total = l_task

        components["total"] = total.item()
        return total, components
