"""
evaluate.py
===========
Evaluation and coefficient analysis for GWF.

Usage:
    python evaluate.py   # loads best model from config.PATHS["best_model_path"]

Outputs:
  - Standard regression metrics: MAE, RMSE, R², MAPE
  - Coefficient analysis:
      * sigma (N, r) spatial map — where regression behaviour differs
      * Effective rank map — complexity of local regression per point
      * Modality importance map — which encoder matters where
  - V2 only: uncertainty map (requires running multiple stochastic passes)
"""

import os
import sys
import logging
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from gwf.models.gwf   import GWF
from gwf.data.dataset import load_gwf_dataset, get_dataloaders
from gwf.data.prompts import PROMPT_TEMPLATES

logger = logging.getLogger("gwf.evaluate")
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s  %(message)s")


# =============================================================================
# Metrics
# =============================================================================

def regression_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """Compute MAE, RMSE, R², MAPE."""
    residuals = y_pred - y_true
    mae  = residuals.abs().mean().item()
    rmse = residuals.pow(2).mean().sqrt().item()
    ss_res = residuals.pow(2).sum()
    ss_tot = (y_true - y_true.mean()).pow(2).sum().clamp(1e-8)
    r2   = (1.0 - ss_res / ss_tot).item()
    # MAPE: skip zero targets
    nonzero = y_true.abs() > 1e-8
    mape = (residuals[nonzero].abs() / y_true[nonzero].abs()).mean().item() * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE%": mape}


def print_metrics(metrics: dict, prefix: str = ""):
    for k, v in metrics.items():
        logger.info(f"  {prefix}{k}: {v:.4f}")


# =============================================================================
# Coefficient analysis
# =============================================================================

def effective_rank(sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute per-point effective rank of the Beta low-rank matrix.
    effective_rank_i = (Σ|sigma_ik|)² / Σ sigma_ik²
    Input : sigma (N, r)
    Output: effective_rank (N,)  ∈ [1, r]
    """
    sigma_abs = sigma.abs()
    numerator   = sigma_abs.sum(-1).pow(2)        # (N,)
    denominator = sigma_abs.pow(2).sum(-1)        # (N,)
    return (numerator / denominator.clamp(1e-8))  # (N,)


def analyse_coefficients(
    coords:       torch.Tensor,      # (N, 2)
    beta:         dict,              # {"U": (N,d,r), "sigma": (N,r), "V": (N,r,d)}
    attn_weights: torch.Tensor,      # (N, 6) modality attention weights
) -> dict:
    """
    Compute interpretable coefficient summaries.

    Returns:
        sigma      : (N, r)  — per-point coefficient magnitudes
        eff_rank   : (N,)    — per-point effective rank ∈ [1, r]
        attn_w     : (N, 6)  — per-point modality importance weights
        coords_np  : (N, 2)  — for spatial plotting
    """
    sigma    = beta["sigma"].cpu()            # (N, r)
    eff_rank = effective_rank(sigma)           # (N,)
    attn_w   = attn_weights.cpu()            # (N, 6)
    coords_np = coords.cpu().numpy()

    return {
        "sigma":      sigma.numpy(),
        "eff_rank":   eff_rank.numpy(),
        "attn_w":     attn_w.numpy(),
        "coords":     coords_np,
    }


def uncertainty_map(
    model:    GWF,
    loader,
    device:   torch.device,
    prompt:   str,
    n_passes: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    V2 only: Monte Carlo uncertainty estimation.
    Runs n_passes stochastic forward passes and computes std of predictions.

    Returns:
        y_mean (N,), y_std (N,)
    """
    model.train()   # enable dropout / stochastic sampling
    all_preds = []

    for _ in range(n_passes):
        preds = []
        with torch.no_grad():
            for batch in loader:
                coords = batch["coords"].to(device)
                X_tab  = batch["X_tab"].to(device)
                y_pred, _ = model(
                    coords=coords, X_tab=X_tab,
                    prompt_text=prompt, deterministic=False,
                )
                preds.append(y_pred.squeeze(-1).cpu())
        all_preds.append(torch.cat(preds))

    stacked = torch.stack(all_preds)   # (n_passes, N)
    return stacked.mean(0), stacked.std(0)


# =============================================================================
# Full evaluation
# =============================================================================

@torch.no_grad()
def full_evaluate(
    model:   GWF,
    loader,
    device:  torch.device,
    prompt:  str,
) -> tuple[dict, dict]:
    """
    Run model on entire loader. Returns metrics dict and analysis dict.
    """
    model.eval()
    all_pred, all_true = [], []
    all_sigma, all_attn = [], []
    all_coords = []

    for batch in loader:
        coords = batch["coords"].to(device)
        X_tab  = batch["X_tab"].to(device)
        y_true = batch["y"].to(device)
        images = batch.get("image")
        if images is not None:
            images = images.to(device)

        y_pred, aux = model(
            coords=coords, X_tab=X_tab,
            images=images, prompt_text=prompt, deterministic=True,
        )
        all_pred.append(y_pred.squeeze(-1).cpu())
        all_true.append(y_true.cpu())
        all_sigma.append(aux["Beta"]["sigma"].cpu())
        all_attn.append(aux["attn_weights"].cpu())
        all_coords.append(coords.cpu())

    y_pred_all  = torch.cat(all_pred)
    y_true_all  = torch.cat(all_true)
    sigma_all   = torch.cat(all_sigma)
    attn_all    = torch.cat(all_attn)
    coords_all  = torch.cat(all_coords)

    metrics  = regression_metrics(y_pred_all, y_true_all)
    analysis = analyse_coefficients(
        coords_all,
        {"sigma": sigma_all},
        attn_all,
    )
    analysis["y_pred"] = y_pred_all.numpy()
    analysis["y_true"] = y_true_all.numpy()

    return metrics, analysis


# =============================================================================
# Main
# =============================================================================

def main():
    cfg_data = config.DATA
    cfg_enc  = config.ENCODERS
    cfg_mod  = config.MODEL
    cfg_path = config.PATHS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Prompt
    prompt = cfg_data.get("prompt_text", "")
    key    = cfg_data.get("prompt_key", "")
    if key and key in PROMPT_TEMPLATES:
        prompt = PROMPT_TEMPLATES[key]
    if not prompt:
        prompt = PROMPT_TEMPLATES["generic"]

    # Dataset
    dataset = load_gwf_dataset(cfg_data, prompt_text=prompt)
    _, val_loader = get_dataloaders(
        dataset,
        val_split=cfg_data["val_split"],
        batch_size=256,
        seed=cfg_data["seed"],
    )

    # Model
    merged_cfg = {**cfg_enc, **cfg_mod,
                  "device": cfg_enc.get("device", "cpu")}
    model = GWF(
        cfg=merged_cfg,
        num_targets=cfg_mod["num_targets"],
        probabilistic=cfg_mod["probabilistic"],
    ).to(device)

    ckpt_path = cfg_path["best_model_path"]
    if not Path(ckpt_path).exists():
        logger.error(f"Checkpoint not found: {ckpt_path}. Run train.py first.")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # Evaluate
    metrics, analysis = full_evaluate(model, val_loader, device, prompt)
    logger.info("\n===== Validation Metrics =====")
    print_metrics(metrics)

    # Coefficient statistics
    sigma = analysis["sigma"]           # (N, r)
    eff_r = analysis["eff_rank"]        # (N,)
    attn  = analysis["attn_w"]          # (N, 6)

    logger.info("\n===== Coefficient Analysis =====")
    logger.info(f"  sigma mean  : {sigma.mean():.4f}  std: {sigma.std():.4f}")
    logger.info(f"  eff_rank    : mean={eff_r.mean():.2f}  "
                f"min={eff_r.min():.2f}  max={eff_r.max():.2f}")

    channel_names = ["SatCLIP", "GeoCLIP", "SkySense", "AnyGraph", "TabPFN", "LLM"]
    logger.info("\n  Mean modality attention weights:")
    for i, name in enumerate(channel_names):
        logger.info(f"    {name:<12}: {attn[:, i].mean():.4f}")

    # Save analysis for plotting
    np.savez(
        "gwf_analysis.npz",
        coords  = analysis["coords"],
        sigma   = sigma,
        eff_rank = eff_r,
        attn_w  = attn,
        y_pred  = analysis["y_pred"],
        y_true  = analysis["y_true"],
    )
    logger.info("\nAnalysis saved to gwf_analysis.npz")
    logger.info("  Use gwf/utils/visualization.py to plot spatial maps.")

    # V2 uncertainty
    if cfg_mod.get("probabilistic", False):
        logger.info("\nComputing uncertainty (V2, 50 MC passes)...")
        y_mean, y_std = uncertainty_map(model, val_loader, device, prompt, n_passes=50)
        logger.info(f"  Uncertainty (std) mean: {y_std.mean():.4f}  max: {y_std.max():.4f}")
        np.savez("gwf_uncertainty.npz",
                 coords=analysis["coords"], y_mean=y_mean.numpy(), y_std=y_std.numpy())
        logger.info("  Saved to gwf_uncertainty.npz")


if __name__ == "__main__":
    main()
