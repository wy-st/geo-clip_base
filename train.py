"""
train.py
========
GWF training script.

Run with:
    python train.py

All hyperparameters are in config.py — edit that file, not this one.

Three-phase training strategy:
  Phase 1  Bridge warm-up   — trains only MLP bridges + CrossChannelFusion
  Phase 2  Joint training   — trains all trainable modules end-to-end
  Phase 3  Task fine-tuning — trains only FiLM + OutputHead (transfer learning)
"""

import os
import sys
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from gwf.models.gwf      import GWF
from gwf.losses.gwf_loss  import GWFLoss
from gwf.data.dataset     import load_gwf_dataset, get_dataloaders
from gwf.data.prompts     import PROMPT_TEMPLATES

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gwf.train")


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    residuals = y_pred - y_true
    rmse = residuals.pow(2).mean().sqrt().item()
    mae  = residuals.abs().mean().item()
    ss_res = residuals.pow(2).sum()
    ss_tot = (y_true - y_true.mean()).pow(2).sum().clamp(min=1e-8)
    r2 = (1.0 - ss_res / ss_tot).item()
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


@torch.no_grad()
def evaluate(model, loader, device, prompt: str) -> dict:
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

    return compute_metrics(torch.cat(preds), torch.cat(trues))


# =============================================================================
# Phase runner
# =============================================================================

def run_phase(
    phase, model, train_loader, val_loader,
    loss_fn, cfg_tr, prompt, device, best_r2, best_path,
) -> float:

    # Phase-specific parameter freeze + LR
    if phase == 1:
        model.freeze_for_phase1()
        lr, epochs = cfg_tr["phase1_lr"], cfg_tr["phase1_epochs"]
    elif phase == 2:
        model.freeze_for_phase2()
        lr, epochs = cfg_tr["phase2_lr"], cfg_tr["phase2_epochs"]
    else:
        model.freeze_for_phase3()
        lr, epochs = cfg_tr["phase3_lr"], cfg_tr["phase3_epochs"]

    if epochs <= 0:
        logger.info(f"Phase {phase} skipped (epochs=0).")
        return best_r2

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nPhase {phase} | {epochs} epochs | lr={lr:.1e} | "
                f"trainable params={trainable:,}")

    opt   = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=cfg_tr["weight_decay"],
    )
    sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))

    # Table header
    print(f"\n{'Phase':^7} {'Epoch':^7} {'Train Loss':^14} "
          f"{'Val RMSE':^11} {'Val R²':^9} {'Time':^8}")
    print("-" * 60)

    for ep in range(1, epochs + 1):
        model.train()
        t0, total_loss, nb = time.time(), 0.0, 0

        for batch in train_loader:
            coords = batch["coords"].to(device)
            X_tab  = batch["X_tab"].to(device)
            y      = batch["y"].to(device)
            images = batch.get("image")
            if images is not None:
                images = images.to(device)

            y_pred, aux = model(
                coords=coords, X_tab=X_tab,
                y=y, images=images, prompt_text=prompt,
            )
            loss, _ = loss_fn(y_pred, y, aux, phase=phase, epoch=ep)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg_tr["grad_clip"],
            )
            opt.step()
            total_loss += loss.item()
            nb += 1

        sched.step()
        avg_loss = total_loss / max(nb, 1)
        elapsed  = time.time() - t0

        metrics = evaluate(model, val_loader, device, prompt)
        r2, rmse = metrics["R2"], metrics["RMSE"]

        star = " ★" if r2 > best_r2 else ""
        print(f"  Ph{phase}    {ep:>3}/{epochs:<3}  "
              f"{avg_loss:>10.4f}    {rmse:>8.4f}    {r2:>7.4f}  "
              f"{elapsed:>5.1f}s{star}")

        if r2 > best_r2:
            best_r2 = r2
            os.makedirs(Path(best_path).parent, exist_ok=True)
            torch.save({
                "phase": phase, "epoch": ep,
                "model_state": model.state_dict(),
                "val_R2": r2, "val_RMSE": rmse,
            }, best_path)

    return best_r2


# =============================================================================
# Main
# =============================================================================

def main():
    cfg_data = config.DATA
    cfg_enc  = config.ENCODERS
    cfg_mod  = config.MODEL
    cfg_loss = config.LOSS
    cfg_tr   = config.TRAINING
    cfg_path = config.PATHS

    # Device
    use_cuda = torch.cuda.is_available() and cfg_tr["device"] == "cuda"
    device   = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Device: {device}")
    torch.manual_seed(cfg_data["seed"])

    # ── Prompt ─────────────────────────────────────────────────────────────
    prompt = cfg_data.get("prompt_text", "")
    key    = cfg_data.get("prompt_key", "")
    if key and key in PROMPT_TEMPLATES:
        prompt = PROMPT_TEMPLATES[key]
    if not prompt:
        prompt = PROMPT_TEMPLATES["generic"]

    # ── Dataset ────────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    dataset = load_gwf_dataset(cfg_data, prompt_text=prompt)
    logger.info(f"Dataset: N={len(dataset)}  features={dataset.X_tab.shape[1]}")

    train_loader, val_loader = get_dataloaders(
        dataset,
        val_split=cfg_data["val_split"],
        batch_size=cfg_tr["batch_size"],
        seed=cfg_data["seed"],
        num_workers=cfg_tr.get("num_workers", 0),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    logger.info("Building GWF model...")
    merged_cfg = {
        **cfg_enc,
        **cfg_mod,
        "device": cfg_enc.get("device", cfg_tr["device"]),
    }
    model = GWF(
        cfg=merged_cfg,
        num_targets=cfg_mod["num_targets"],
        probabilistic=cfg_mod["probabilistic"],
    ).to(device)

    tot = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {tot:,}")

    # ── Loss ───────────────────────────────────────────────────────────────
    loss_fn = GWFLoss(
        cfg=cfg_loss,
        task=cfg_mod["task"],
        probabilistic=cfg_mod["probabilistic"],
    )

    best_path = cfg_path["best_model_path"]
    best_r2   = float("-inf")

    # ── Training ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: Bridge Warm-up")
    best_r2 = run_phase(1, model, train_loader, val_loader,
                        loss_fn, cfg_tr, prompt, device, best_r2, best_path)

    logger.info("=" * 60)
    logger.info("PHASE 2: Joint Training")
    best_r2 = run_phase(2, model, train_loader, val_loader,
                        loss_fn, cfg_tr, prompt, device, best_r2, best_path)

    if cfg_tr.get("phase3_epochs", 0) > 0:
        logger.info("=" * 60)
        logger.info("PHASE 3: Task Fine-tuning")
        best_r2 = run_phase(3, model, train_loader, val_loader,
                            loss_fn, cfg_tr, prompt, device, best_r2, best_path)

    logger.info("=" * 60)
    logger.info(f"Training complete.  Best val R² = {best_r2:.4f}")
    logger.info(f"Best model saved → {best_path}")


if __name__ == "__main__":
    main()
