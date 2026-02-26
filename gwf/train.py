"""
GWF — Training & Evaluation Script

Usage (from repo root):
    python -m gwf.train                         # synthetic data, default config
    python -m gwf.train --source california     # California Housing
    python -m gwf.train --epochs 300 --lr 5e-4  # custom hyper-params
    python -m gwf.train --visualise             # save β t-SNE map after training
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── project imports ────────────────────────────────────────────────────────────
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from gwf.config import GWFConfig
from gwf.data   import get_dataloaders
from gwf.model  import GWF


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return math.sqrt(((y_hat - y) ** 2).mean().item())


def r2(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    ss_res = ((y - y_hat) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    return 1.0 - ss_res / (ss_tot + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: GWFConfig, source: str = "synthetic",
          visualise: bool = False, device: str = "cpu",
          csv_path: str | None = None,
          lat_col: str = "lat", lon_col: str = "lon",
          target_col: str = "price", feature_cols: list | None = None):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"\n{'─'*60}")
    print(f"  GWF  |  source={source}  |  device={device}")
    print(f"{'─'*60}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_dl, val_dl, feat_dim = get_dataloaders(
        source=source,
        k=cfg.k_neighbors,
        val_split=cfg.val_split,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        n_samples=cfg.n_samples,
        csv_path=csv_path,
        lat_col=lat_col,
        lon_col=lon_col,
        target_col=target_col,
        feature_cols=feature_cols)

    cfg.feat_dim = feat_dim
    print(f"  feat_dim={feat_dim}  |  k={cfg.k_neighbors}  |  "
          f"train={len(train_dl.dataset)}  val={len(val_dl.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = GWF(
        feat_dim     = feat_dim,
        loc_proj_dim = cfg.loc_proj_dim,
        node_dim     = cfg.node_dim,
        z_proj_dim   = cfg.z_proj_dim,
        kernel_rank  = cfg.kernel_rank,
        attn_dim     = cfg.attn_dim,
        wls_lambda   = cfg.wls_lambda,
        tabpfn_path  = cfg.tabpfn_path,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    opt  = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state    = None

    print(f"\n{'Epoch':>6}  {'Train-RMSE':>11}  {'Val-RMSE':>9}  "
          f"{'Val-R²':>7}  {'Time':>6}")
    print("─" * 52)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            y_hat, _ = model(batch)
            loss = loss_fn(y_hat, batch["y"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_losses.append(loss.item())

        sched.step()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_yhat, val_ytrue = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}
                yh, _ = model(batch)
                val_yhat.append(yh.cpu())
                val_ytrue.append(batch["y"].cpu())

        val_yhat  = torch.cat(val_yhat)
        val_ytrue = torch.cat(val_ytrue)

        tr_rmse  = math.sqrt(np.mean(train_losses))
        vl_rmse  = rmse(val_yhat, val_ytrue)
        vl_r2    = r2(val_yhat, val_ytrue)
        elapsed  = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_rmse:>11.4f}  {vl_rmse:>9.4f}  "
                  f"{vl_r2:>7.4f}  {elapsed:>5.1f}s")

        if vl_rmse < best_val_rmse:
            best_val_rmse = vl_rmse
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\n  Best val RMSE: {best_val_rmse:.4f}")

    # ── Restore best weights ───────────────────────────────────────────────
    model.load_state_dict(best_state)

    # ── Optional: β t-SNE visualisation ───────────────────────────────────
    if visualise:
        _visualise_betas(model, val_dl, device, source)

    return model, best_val_rmse


# ──────────────────────────────────────────────────────────────────────────────
# β t-SNE visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _visualise_betas(model, val_dl, device, source):
    """
    Project local GWR coefficients β_i ∈ R^p to 1-D via t-SNE,
    then plot them on the spatial map coloured by the t-SNE value.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [visualise] matplotlib / sklearn not available — skipping")
        return

    print("\n  Running t-SNE on local β coefficients …")
    y_hat, betas, coords, y_true = model.predict_with_betas(val_dl, device)

    betas_np = betas.numpy()              # (N_val, p)
    coords_np = coords.numpy()            # (N_val, 2)

    # t-SNE to 1-D
    tsne = TSNE(n_components=1, perplexity=min(30, len(betas_np) - 1),
                random_state=42, n_iter=500)
    beta_1d = tsne.fit_transform(betas_np).squeeze()   # (N_val,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: β t-SNE (spatial regime map)
    sc0 = axes[0].scatter(coords_np[:, 1], coords_np[:, 0],
                           c=beta_1d, cmap="RdYlBu", s=8, alpha=0.7)
    plt.colorbar(sc0, ax=axes[0], label="β t-SNE (1-D)")
    axes[0].set_title("Spatial Regimes — local GWR β projected to 1-D")
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

    # Right: prediction vs truth
    axes[1].scatter(y_true.numpy(), y_hat.numpy(), s=5, alpha=0.4)
    lim = [min(y_true.min(), y_hat.min()).item(),
           max(y_true.max(), y_hat.max()).item()]
    axes[1].plot(lim, lim, "r--", linewidth=1)
    axes[1].set_title("Predicted vs True")
    axes[1].set_xlabel("y_true"); axes[1].set_ylabel("y_hat")

    out = f"gwf_beta_tsne_{source}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"  Saved: {out}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        "GWF Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m gwf.train                                      # synthetic data
  python -m gwf.train --source california                  # CA Housing
  python -m gwf.train --source custom --csv housing.csv \\
      --lat_col lat --lon_col lon --target price           # your own CSV
  python -m gwf.train --source custom --csv housing.csv \\
      --features area rooms age dist_subway --visualise    # select features
""")
    parser.add_argument("--source",    default="synthetic",
                        choices=["synthetic", "california", "custom"])
    # ── Custom dataset ────────────────────────────────────────────────────
    parser.add_argument("--csv",        default=None,
                        metavar="PATH",  help="path to your CSV file")
    parser.add_argument("--lat_col",    default="lat",
                        help="latitude column name  (default: lat)")
    parser.add_argument("--lon_col",    default="lon",
                        help="longitude column name (default: lon)")
    parser.add_argument("--target",     default="price",
                        metavar="COL",  help="target column name (default: price)")
    parser.add_argument("--features",   nargs="+", default=None,
                        metavar="COL",  help="feature columns (default: all except lat/lon/target)")
    # ── Training hyper-params ─────────────────────────────────────────────
    parser.add_argument("--epochs",    type=int,   default=None)
    parser.add_argument("--lr",        type=float, default=None)
    parser.add_argument("--batch",     type=int,   default=None)
    parser.add_argument("--k",         type=int,   default=None,
                        help="k-NN neighbours")
    parser.add_argument("--n_samples", type=int,   default=None,
                        help="synthetic dataset size")
    parser.add_argument("--tabpfn_path", default=None,
                        metavar="PATH",
                        help="path to local TabPFN regressor .ckpt; "
                             "None → attempt HuggingFace download")
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    cfg = GWFConfig()
    if args.epochs      is not None: cfg.epochs      = args.epochs
    if args.lr          is not None: cfg.lr          = args.lr
    if args.batch       is not None: cfg.batch_size  = args.batch
    if args.k           is not None: cfg.k_neighbors = args.k
    if args.n_samples   is not None: cfg.n_samples   = args.n_samples
    if args.tabpfn_path is not None: cfg.tabpfn_path = args.tabpfn_path

    train(cfg,
          source=args.source,
          visualise=args.visualise,
          device=args.device,
          csv_path=args.csv,
          lat_col=args.lat_col,
          lon_col=args.lon_col,
          target_col=args.target,
          feature_cols=args.features)


if __name__ == "__main__":
    main()
