"""
gwf/utils/transfer.py
======================
Cross-region zero-shot and few-shot transfer evaluation utilities.

GWF can transfer across geographic regions because:
  1. Foundation models encode universal geographic semantics
  2. The LLM channel steers task-specific behaviour via prompt
  3. The spatial GNN generalises over any coordinate space

Usage:
    from gwf.utils.transfer import zero_shot_eval, few_shot_finetune

    # Zero-shot: use Phase 2 model directly on a new region
    metrics = zero_shot_eval(model, target_loader, device, prompt)

    # Few-shot: fine-tune only the output head + FiLM on a small target set
    model = few_shot_finetune(model, few_shot_loader, device, prompt, epochs=10)
"""

import logging
import torch
import torch.nn as nn
from torch.optim import AdamW

logger = logging.getLogger("gwf.transfer")


# =============================================================================
# Zero-shot evaluation
# =============================================================================

@torch.no_grad()
def zero_shot_eval(
    model,
    loader,
    device:  torch.device,
    prompt:  str,
) -> dict[str, float]:
    """
    Evaluate a trained GWF model on a new geographic region WITHOUT any
    fine-tuning (Phase 3 is skipped). The model relies entirely on the
    generalisation of its Phase 2 weights + the task prompt.

    Args:
        model   : trained GWF model (Phase 2 complete)
        loader  : DataLoader for the target region
        device  : compute device
        prompt  : task description prompt for the target region/dataset

    Returns:
        dict with MAE, RMSE, R², MAPE
    """
    model.eval()
    preds, trues = [], []

    for batch in loader:
        coords = batch["coords"].to(device)
        X_tab  = batch["X_tab"].to(device)
        y_true = batch["y"].to(device)
        images = batch.get("image")
        if images is not None:
            images = images.to(device)

        y_pred, _ = model(
            coords=coords, X_tab=X_tab,
            images=images, prompt_text=prompt, deterministic=True,
        )
        preds.append(y_pred.squeeze(-1).cpu())
        trues.append(y_true.cpu())

    y_pred_all = torch.cat(preds)
    y_true_all = torch.cat(trues)

    residuals = y_pred_all - y_true_all
    mae  = residuals.abs().mean().item()
    rmse = residuals.pow(2).mean().sqrt().item()
    ss_res = residuals.pow(2).sum()
    ss_tot = (y_true_all - y_true_all.mean()).pow(2).sum().clamp(1e-8)
    r2   = (1.0 - ss_res / ss_tot).item()
    nonzero = y_true_all.abs() > 1e-8
    mape = (residuals[nonzero].abs() / y_true_all[nonzero].abs()).mean().item() * 100

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE%": mape}
    logger.info(f"Zero-shot eval → R²={r2:.4f}  RMSE={rmse:.4f}")
    return metrics


# =============================================================================
# Few-shot fine-tuning (Phase 3 on target data)
# =============================================================================

def few_shot_finetune(
    model,
    few_shot_loader,
    device:      torch.device,
    prompt:      str,
    epochs:      int   = 10,
    lr:          float = 1e-4,
    weight_decay: float = 0.01,
    grad_clip:   float = 1.0,
) -> object:
    """
    Phase 3 fine-tuning on a small target dataset.

    Freezes everything EXCEPT FiLM + OutputHead, then trains for a few epochs
    on the target region data. This is fast since very few parameters update.

    Args:
        model            : GWF model with Phase 2 weights
        few_shot_loader  : DataLoader for the small target dataset
        device           : compute device
        prompt           : task prompt for the target region
        epochs           : number of fine-tuning epochs (default 10)
        lr               : learning rate for Phase 3 (default 1e-4)
        weight_decay     : weight decay (default 0.01)
        grad_clip        : gradient clipping max norm (default 1.0)

    Returns:
        Fine-tuned model (same object, modified in-place).
    """
    import torch.nn.functional as F

    model.freeze_for_phase3()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Few-shot fine-tuning | {epochs} epochs | "
                f"trainable params={trainable:,}")

    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )

    for ep in range(1, epochs + 1):
        model.train()
        total, nb = 0.0, 0

        for batch in few_shot_loader:
            coords = batch["coords"].to(device)
            X_tab  = batch["X_tab"].to(device)
            y      = batch["y"].to(device)
            images = batch.get("image")
            if images is not None:
                images = images.to(device)

            y_pred, _ = model(
                coords=coords, X_tab=X_tab,
                images=images, y=y, prompt_text=prompt,
            )
            loss = F.mse_loss(y_pred.squeeze(-1), y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )
            opt.step()
            total += loss.item()
            nb    += 1

        logger.info(f"  Epoch {ep}/{epochs}  loss={total/max(nb,1):.4f}")

    return model


# =============================================================================
# Multi-task dataset iterator
# =============================================================================

class MultiTaskIterator:
    """
    Cycles through multiple dataset DataLoaders, alternating one batch at a time.
    Used for multi-task Phase 2 training with multiple geographic datasets.

    Each dataset has its own prompt_text describing the task and region.

    Args:
        loaders      : list of DataLoader objects
        prompt_texts : list of prompt strings (one per loader)

    Usage:
        mt = MultiTaskIterator([loader_pm25, loader_housing], [prompt_pm25, prompt_housing])
        for batch, prompt in mt:
            y_pred, aux = model(... prompt_text=prompt ...)
    """

    def __init__(self, loaders: list, prompt_texts: list[str]):
        assert len(loaders) == len(prompt_texts), \
            "loaders and prompt_texts must have the same length."
        self.loaders      = loaders
        self.prompt_texts = prompt_texts
        self._iters       = [iter(dl) for dl in loaders]
        self._idx         = 0

    def __iter__(self):
        self._iters = [iter(dl) for dl in self.loaders]
        self._idx   = 0
        return self

    def __next__(self):
        # Round-robin through datasets
        n = len(self.loaders)
        for _ in range(n):
            i = self._idx % n
            self._idx += 1
            try:
                batch = next(self._iters[i])
                return batch, self.prompt_texts[i]
            except StopIteration:
                self._iters[i] = iter(self.loaders[i])
                batch = next(self._iters[i])
                return batch, self.prompt_texts[i]
        raise StopIteration
