"""
gwf/losses/kl_loss.py
======================
KL Divergence Loss (L_KL) — V2 only

Regularises the Beta posterior distributions in the variational HyperNet.
The KL loss is already computed inside HyperNetBeta.forward() and stored
in beta["kl_loss"]. This module simply extracts it with KL annealing.

KL Annealing:
  During the first kl_annealing_epochs epochs of Phase 2, λ_KL is ramped
  up from 0 → lambda_kl to prevent early KL domination.
"""

import torch
import torch.nn as nn


class KLLoss(nn.Module):
    """
    Wrapper that extracts the precomputed KL from the Beta dict and applies
    KL annealing based on current epoch.

    Args:
        lambda_kl            : target KL weight (default 0.01)
        kl_annealing_epochs  : ramp-up duration in epochs (default 20)
    """

    def __init__(
        self,
        lambda_kl:           float = 0.01,
        kl_annealing_epochs: int   = 20,
    ):
        super().__init__()
        self.lambda_kl           = lambda_kl
        self.kl_annealing_epochs = kl_annealing_epochs

    def effective_weight(self, epoch: int) -> float:
        """Returns the effective λ_KL at a given epoch."""
        if self.kl_annealing_epochs <= 0:
            return self.lambda_kl
        ramp = min(epoch / self.kl_annealing_epochs, 1.0)
        return ramp * self.lambda_kl

    def forward(
        self,
        beta:  dict[str, torch.Tensor],
        epoch: int = 999,
    ) -> torch.Tensor:
        """
        beta  : must contain "kl_loss" key (set by HyperNetBeta V2 during training)
        epoch : current training epoch (for KL annealing)
        Returns scalar KL loss weighted by effective λ.
        """
        kl = beta.get("kl_loss", torch.tensor(0.0))
        w  = self.effective_weight(epoch)
        return w * kl
