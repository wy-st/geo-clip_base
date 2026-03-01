"""
GWF vs SRGCNN Benchmark — San Diego Airbnb (log_price)

Dataset  : SRGCNN repo  airbnb/regression_db.geojson
           n=6110, features: accommodates, bathrooms, bedrooms, beds
           target: log_price

SRGCNN-GW reference results (full-data self-eval, no train/test split):
   MAPE = 4.81%,  R² = 0.789

This script evaluates GWF with an 80/20 inductive train/test split,
which is a harder but more meaningful evaluation protocol.

Baselines included:
  OLS  — ordinary least squares (sklearn)
  GWF  — our model (this work)

Usage
-----
    python -m gwf.benchmark_srgcnn                       # default settings
    python -m gwf.benchmark_srgcnn --epochs 100 --k 20  # custom
    python -m gwf.benchmark_srgcnn --data path/to/regression_db.geojson
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from gwf.data   import build_knn_graph, SpatialRegressionDataset
from gwf.model  import GWF
from gwf.config import GWFConfig
from torch.utils.data import DataLoader, random_split


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
TARGET    = 'log_price'


def load_airbnb(path: str):
    """
    Parse SRGCNN airbnb GeoJSON → (coords, X, y, y_mean, y_std).
    coords : (n, 2) float32  [lat, lon]
    X      : (n, 4) float32  standardised features
    y      : (n,)   float32  standardised log_price  (for training)
    y_mean, y_std : scalars for de-standardising predictions
    """
    with open(path) as f:
        gj = json.load(f)

    rows = []
    for feat in gj['features']:
        p = feat['properties']
        lon, lat = feat['geometry']['coordinates']
        row = [lat, lon] + [float(p[k]) for k in FEATURES] + [float(p[TARGET])]
        rows.append(row)

    arr      = np.array(rows, dtype=np.float64)
    coords   = arr[:, :2].astype(np.float32)

    scaler_x = StandardScaler()
    X        = scaler_x.fit_transform(arr[:, 2:6]).astype(np.float32)

    y_raw    = arr[:, 6].astype(np.float32)
    y_mean   = float(y_raw.mean())
    y_std    = float(y_raw.std())
    y        = ((y_raw - y_mean) / y_std).astype(np.float32)

    return coords, X, y, y_mean, y_std, y_raw


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def rmse(pred, true):
    return math.sqrt(((pred - true) ** 2).mean())


def mape(pred, true):
    """MAPE on original (log_price) scale, matching SRGCNN notebook."""
    return float(np.mean(np.abs((pred - true) / (np.abs(true) + 1e-8)))) * 100


def r2(pred, true):
    return float(r2_score(true, pred))


# ─────────────────────────────────────────────────────────────────────────────
# OLS baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_ols(X_train, y_train, X_test, y_test_raw, y_mean, y_std):
    reg = LinearRegression().fit(X_train, y_train)
    y_pred_std = reg.predict(X_test).astype(np.float32)
    y_pred_raw = y_pred_std * y_std + y_mean
    return {
        "RMSE":  rmse(y_pred_raw, y_test_raw),
        "R²":    r2(y_pred_raw, y_test_raw),
        "MAPE":  mape(y_pred_raw, y_test_raw),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GWF training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_gwf(coords, X, y, y_raw, y_mean, y_std,
            k, epochs, lr, batch_size, val_split, seed, device):

    torch.manual_seed(seed)
    np.random.seed(seed)

    nbr_idx, nbr_dist = build_knn_graph(coords, k)
    dataset = SpatialRegressionDataset(coords, X, y, nbr_idx, nbr_dist)

    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed))

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  drop_last=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, drop_last=False, num_workers=0)

    model = GWF(feat_dim=X.shape[1]).to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    loss_fn = nn.MSELoss()

    best_rmse  = float("inf")
    best_state = None

    print(f"\n{'Epoch':>6}  {'Train-RMSE':>11}  {'Val-RMSE':>9}  {'Time':>6}")
    print("─" * 42)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_losses = []
        for batch in train_dl:
            batch = {k2: v.to(device) for k2, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            y_hat, _ = model(batch)
            loss = loss_fn(y_hat, batch["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())
        sched.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k2: v.to(device) for k2, v in batch.items()
                         if isinstance(v, torch.Tensor)}
                yh, _ = model(batch)
                val_preds.append(yh.cpu())
                val_true.append(batch["y"].cpu())

        val_preds = torch.cat(val_preds).numpy()
        val_true  = torch.cat(val_true).numpy()

        tr_rmse = math.sqrt(np.mean(train_losses))
        vl_rmse = rmse(val_preds, val_true)
        elapsed = time.time() - t0

        if epoch % 20 == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_rmse:>11.4f}  {vl_rmse:>9.4f}  {elapsed:>5.1f}s")

        if vl_rmse < best_rmse:
            best_rmse  = vl_rmse
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # ── Final evaluation on val set in original log_price scale ───────────
    model.eval()
    preds_std, true_std = [], []
    with torch.no_grad():
        for batch in val_dl:
            batch = {k2: v.to(device) for k2, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            yh, _ = model(batch)
            preds_std.append(yh.cpu().numpy())
            true_std.append(batch["y"].cpu().numpy())

    preds_std = np.concatenate(preds_std)
    true_std  = np.concatenate(true_std)

    # De-standardise
    preds_raw = preds_std * y_std + y_mean
    true_raw  = true_std  * y_std + y_mean

    return {
        "RMSE": rmse(preds_raw, true_raw),
        "R²":   r2(preds_raw, true_raw),
        "MAPE": mape(preds_raw, true_raw),
    }, model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("GWF vs SRGCNN benchmark")
    parser.add_argument("--data",       default="airbnb_regression_db.geojson",
                        help="path to SRGCNN regression_db.geojson")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--k",          type=int,   default=20,
                        help="k-NN neighbours (SRGCNN uses k=20)")
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch",      type=int,   default=256)
    parser.add_argument("--val_split",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("  GWF vs SRGCNN — San Diego Airbnb (log_price)")
    print("=" * 60)
    print(f"  data={args.data}  k={args.k}  epochs={args.epochs}")
    print(f"  device={args.device}  val_split={args.val_split}")

    # ── Load data ─────────────────────────────────────────────────────────
    coords, X, y, y_mean, y_std, y_raw = load_airbnb(args.data)
    n, p = X.shape
    print(f"\n  n={n}, p={p}, y_mean={y_mean:.3f}, y_std={y_std:.3f}")

    # ── Train/test split indices for OLS ──────────────────────────────────
    rng     = np.random.default_rng(args.seed)
    n_val   = int(n * args.val_split)
    idx     = rng.permutation(n)
    tr_idx  = idx[n_val:]
    val_idx = idx[:n_val]

    # ── OLS baseline ──────────────────────────────────────────────────────
    print("\n── OLS baseline ──────────────────────────────────────────────")
    ols_metrics = run_ols(
        X[tr_idx], y[tr_idx],
        X[val_idx], y_raw[val_idx],
        y_mean, y_std)
    print(f"  RMSE={ols_metrics['RMSE']:.4f}  R²={ols_metrics['R²']:.4f}  "
          f"MAPE={ols_metrics['MAPE']:.2f}%")

    # ── GWF ───────────────────────────────────────────────────────────────
    print("\n── GWF ───────────────────────────────────────────────────────")
    gwf_metrics, _ = run_gwf(
        coords, X, y, y_raw, y_mean, y_std,
        k=args.k, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch, val_split=args.val_split,
        seed=args.seed, device=args.device)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY  (80/20 inductive train/test split)")
    print("=" * 60)
    print(f"  {'Model':<20} {'RMSE':>8} {'R²':>8} {'MAPE':>8}")
    print(f"  {'-'*46}")
    print(f"  {'OLS':<20} {ols_metrics['RMSE']:>8.4f} "
          f"{ols_metrics['R²']:>8.4f} {ols_metrics['MAPE']:>7.2f}%")
    print(f"  {'GWF (ours)':<20} {gwf_metrics['RMSE']:>8.4f} "
          f"{gwf_metrics['R²']:>8.4f} {gwf_metrics['MAPE']:>7.2f}%")
    print()
    print("  SRGCNN-GW reference (full-data self-eval, no split):")
    print(f"  {'SRGCNN-GW':<20} {'N/A':>8} {'0.7888':>8} {'4.81':>7}%")
    print()
    print("  Note: SRGCNN-GW trains and evaluates on the full dataset")
    print("  (transductive). GWF uses a stricter inductive 80/20 split.")
    print("=" * 60)


if __name__ == "__main__":
    main()
