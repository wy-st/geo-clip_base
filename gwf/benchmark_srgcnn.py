"""
GWF vs SRGCNN-GW vs OLS — San Diego Airbnb  (log_price)
Unified 6:2:2 random train/val/test split, same seed for all models.

Dataset : airbnb/regression_db.geojson  (n=6110)
Features: accommodates, bathrooms, bedrooms, beds  (p=4)
Target  : log_price

Usage
-----
    python -m gwf.benchmark_srgcnn
    python -m gwf.benchmark_srgcnn --gwf_epochs 100 --srgcnn_epochs 5000
    python -m gwf.benchmark_srgcnn --data path/to/regression_db.geojson
"""

import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags, eye as speye
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from gwf.data   import build_knn_graph, SpatialRegressionDataset
from gwf.model  import GWF

FEATURES = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
TARGET   = 'log_price'


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_airbnb(path):
    with open(path) as f:
        gj = json.load(f)
    rows = []
    for feat in gj['features']:
        p = feat['properties']
        lon, lat = feat['geometry']['coordinates']
        rows.append([lat, lon] + [float(p[k]) for k in FEATURES] + [float(p[TARGET])])
    arr    = np.array(rows, dtype=np.float64)
    coords = arr[:, :2].astype(np.float32)
    scaler = StandardScaler()
    X      = scaler.fit_transform(arr[:, 2:6]).astype(np.float32)
    y_raw  = arr[:, 6].astype(np.float32)
    y_mean, y_std = float(y_raw.mean()), float(y_raw.std())
    y_norm = ((y_raw - y_mean) / y_std).astype(np.float32)
    return coords, X, y_norm, y_raw, y_mean, y_std


def split_622(n, seed=42):
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(n)
    n_te = int(n * 0.2)
    n_va = int(n * 0.2)
    return idx[n_te + n_va:], idx[n_te:n_te + n_va], idx[:n_te]   # tr, val, te


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def metrics(pred, true):
    rmse = math.sqrt(((pred - true) ** 2).mean())
    r2   = float(r2_score(true, pred))
    mape = float(np.mean(np.abs((pred - true) / (np.abs(true) + 1e-8)))) * 100
    return rmse, r2, mape


# ─────────────────────────────────────────────────────────────────────────────
# OLS
# ─────────────────────────────────────────────────────────────────────────────

def run_ols(X, y_norm, y_raw, y_mean, y_std, tr, te):
    reg = LinearRegression().fit(X[tr], y_norm[tr])
    pred_raw = reg.predict(X[te]).astype(np.float32) * y_std + y_mean
    return metrics(pred_raw, y_raw[te])


# ─────────────────────────────────────────────────────────────────────────────
# SRGCNN-GW  (reproduced from dizhu-gis/SRGCNN notebook)
# ─────────────────────────────────────────────────────────────────────────────

def _build_adj(coords, k=20):
    """Symmetric k-NN Laplacian normalised adjacency (renorm trick)."""
    nn_obj = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
    _, idx = nn_obj.kneighbors(coords)
    n = len(coords)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    data = np.ones(len(rows))
    A    = csr_matrix((data, (rows, cols)), shape=(n, n)).toarray()
    A    = np.logical_or(A, A.T).astype(float)   # symmetrise
    A    = A + np.eye(n)                          # add self-loops (renorm trick)
    deg  = A.sum(1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return d_inv_sqrt @ A @ d_inv_sqrt            # D^{-1/2} A~ D^{-1/2}


class GWGraphConvolution(nn.Module):
    def __init__(self, n_nodes, f_in, f_out, activation=nn.ReLU()):
        super().__init__()
        self.activation  = activation
        self.gwr_weight  = nn.Parameter(torch.ones(n_nodes, f_in))
        self.weight      = nn.Parameter(torch.ones(f_in, f_out))
        self.bias        = nn.Parameter(torch.zeros(f_out))

    def forward(self, x, adj):
        out = torch.mm(adj, torch.mul(x, self.gwr_weight))
        out = torch.mm(out, self.weight) + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out


class GWGCN(nn.Module):
    def __init__(self, n_nodes, f_in, f_out, hidden, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        dims = [f_in] + hidden
        self.layers = nn.ModuleList([
            GWGraphConvolution(n_nodes, dims[i], dims[i+1])
            for i in range(len(hidden))
        ])
        self.out_layer = GWGraphConvolution(n_nodes, dims[-1], f_out,
                                            activation=None)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)
        return self.out_layer(x, adj)


def run_srgcnn_gw(X, y_raw, y_mean, y_std, coords, tr, va, te,
                  epochs, lr, k, device):
    n = len(X)
    print(f"\n  Building {n}×{n} adjacency (k={k})…", end=" ", flush=True)
    adj_np = _build_adj(coords, k=k)
    print("done")

    adj  = torch.FloatTensor(adj_np).to(device)
    y_norm = ((y_raw - y_mean) / y_std).astype(np.float32)

    x_t = torch.FloatTensor(X).to(device)
    y_t = torch.FloatTensor(y_norm).unsqueeze(1).to(device)

    tr_idx = torch.LongTensor(tr).to(device)
    va_idx = torch.LongTensor(va).to(device)
    te_idx = torch.LongTensor(te).to(device)

    f_in   = X.shape[1]
    hidden = [8 * f_in]   # = [32] matching notebook default

    model = GWGCN(n, f_in, 1, hidden, dropout=0.5).to(device)
    opt   = Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None

    print(f"\n{'Epoch':>7}  {'TrainMSE':>9}  {'ValMSE':>8}  {'Time':>6}")
    print("─" * 38)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        out  = model(x_t, adj)
        loss = F.mse_loss(out[tr_idx], y_t[tr_idx])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_val = model(x_t, adj)
            val_mse = F.mse_loss(out_val[va_idx], y_t[va_idx]).item()

        if val_mse < best_val:
            best_val   = val_mse
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            print(f"{ep:>7}  {loss.item():>9.4f}  {val_mse:>8.4f}  "
                  f"{time.time()-t0:>5.1f}s")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_te  = model(x_t, adj)[te_idx].cpu().numpy().ravel()
    pred_raw = out_te * y_std + y_mean
    return metrics(pred_raw, y_raw[te])


# ─────────────────────────────────────────────────────────────────────────────
# GWF
# ─────────────────────────────────────────────────────────────────────────────

def run_gwf(coords, X, y_norm, y_raw, y_mean, y_std,
            tr, va, te, epochs, lr, k, batch_size, seed, device):

    torch.manual_seed(seed)
    nbr_idx, nbr_dist = build_knn_graph(coords, k)
    dataset = SpatialRegressionDataset(coords, X, y_norm, nbr_idx, nbr_dist)

    train_dl = DataLoader(Subset(dataset, tr), batch_size=batch_size,
                          shuffle=True, drop_last=True, num_workers=0)
    val_dl   = DataLoader(Subset(dataset, va), batch_size=batch_size,
                          shuffle=False, num_workers=0)
    test_dl  = DataLoader(Subset(dataset, te), batch_size=batch_size,
                          shuffle=False, num_workers=0)

    model   = GWF(feat_dim=X.shape[1]).to(device)
    opt     = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    loss_fn = nn.MSELoss()

    best_val  = float("inf")
    best_state = None

    print(f"\n{'Epoch':>6}  {'TrainRMSE':>10}  {'ValRMSE':>8}  {'Time':>6}")
    print("─" * 40)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses = []
        for batch in train_dl:
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            yh, _ = model(batch)
            loss  = loss_fn(yh, batch["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())
        sched.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {kk: vv.to(device) for kk, vv in batch.items()
                         if isinstance(vv, torch.Tensor)}
                yh, _ = model(batch)
                val_preds.append(yh.cpu()); val_true.append(batch["y"].cpu())
        val_rmse = math.sqrt(np.mean((torch.cat(val_preds).numpy() -
                                      torch.cat(val_true).numpy()) ** 2))

        if val_rmse < best_val:
            best_val   = val_rmse
            best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            tr_rmse = math.sqrt(np.mean(tr_losses))
            print(f"{ep:>6}  {tr_rmse:>10.4f}  {val_rmse:>8.4f}  "
                  f"{time.time()-t0:>5.1f}s")

    model.load_state_dict(best_state)
    model.eval()
    te_preds, te_true = [], []
    with torch.no_grad():
        for batch in test_dl:
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            yh, _ = model(batch)
            te_preds.append(yh.cpu()); te_true.append(batch["y"].cpu())

    pred_norm = torch.cat(te_preds).numpy()
    true_norm = torch.cat(te_true).numpy()
    pred_raw  = pred_norm * y_std + y_mean
    true_raw  = true_norm * y_std + y_mean
    return metrics(pred_raw, true_raw)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",          default="airbnb_regression_db.geojson")
    ap.add_argument("--k",             type=int,   default=20)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--gwf_epochs",    type=int,   default=50)
    ap.add_argument("--gwf_lr",        type=float, default=3e-4)
    ap.add_argument("--gwf_batch",     type=int,   default=256)
    ap.add_argument("--srgcnn_epochs", type=int,   default=3000)
    ap.add_argument("--srgcnn_lr",     type=float, default=3e-2)
    ap.add_argument("--device",        default="cpu")
    args = ap.parse_args()

    print("=" * 62)
    print("  GWF vs SRGCNN-GW vs OLS — San Diego Airbnb  (log_price)")
    print(f"  Split: 60% train / 20% val / 20% test  (seed={args.seed})")
    print("=" * 62)

    coords, X, y_norm, y_raw, y_mean, y_std = load_airbnb(args.data)
    n = len(X)
    tr, va, te = split_622(n, args.seed)
    print(f"  n={n}  train={len(tr)}  val={len(va)}  test={len(te)}  p={X.shape[1]}")

    results = {}

    # ── OLS ───────────────────────────────────────────────────────────────
    print("\n── OLS ───────────────────────────────────────────────────────")
    results["OLS"] = run_ols(X, y_norm, y_raw, y_mean, y_std, tr, te)
    rmse, r2, mape = results["OLS"]
    print(f"  RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    # ── SRGCNN-GW ─────────────────────────────────────────────────────────
    print(f"\n── SRGCNN-GW  ({args.srgcnn_epochs} epochs, lr={args.srgcnn_lr}) ──")
    results["SRGCNN-GW"] = run_srgcnn_gw(
        X, y_raw, y_mean, y_std, coords, tr, va, te,
        epochs=args.srgcnn_epochs, lr=args.srgcnn_lr,
        k=args.k, device=args.device)
    rmse, r2, mape = results["SRGCNN-GW"]
    print(f"\n  Test → RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    # ── GWF ───────────────────────────────────────────────────────────────
    print(f"\n── GWF  ({args.gwf_epochs} epochs, lr={args.gwf_lr}) ────────────")
    results["GWF"] = run_gwf(
        coords, X, y_norm, y_raw, y_mean, y_std,
        tr, va, te,
        epochs=args.gwf_epochs, lr=args.gwf_lr,
        k=args.k, batch_size=args.gwf_batch,
        seed=args.seed, device=args.device)
    rmse, r2, mape = results["GWF"]
    print(f"\n  Test → RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FINAL RESULTS — 20% held-out test set")
    print("=" * 62)
    print(f"  {'Model':<18} {'RMSE':>8} {'R²':>8} {'MAPE':>8}")
    print(f"  {'-'*46}")
    for name, (rmse, r2, mape) in results.items():
        print(f"  {name:<18} {rmse:>8.4f} {r2:>8.4f} {mape:>7.2f}%")
    print("=" * 62)


if __name__ == "__main__":
    main()
