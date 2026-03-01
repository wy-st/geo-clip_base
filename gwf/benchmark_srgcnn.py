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

from gwf.data    import build_knn_graph, SpatialRegressionDataset
from gwf.model   import GWF
from gwf.model_u import GWF_U

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


def split_spatial_622(coords):
    """Geographic band split sorted by latitude (S→N).

    Keeps 6:2:2 ratio but with spatially contiguous train / val / test regions
    so that test nodes are geographically separated from training nodes.
    """
    order  = np.argsort(coords[:, 0])    # sort S→N by latitude
    n      = len(order)
    n_te   = int(n * 0.2)
    n_va   = int(n * 0.2)
    tr_idx = order[:n - n_te - n_va]     # southernmost 60% → train
    va_idx = order[n - n_te - n_va: n - n_te]
    te_idx = order[n - n_te:]            # northernmost 20% → test
    return tr_idx, va_idx, te_idx


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
# Classical GWR  (analytical, adaptive k-NN bandwidth, no epochs needed)
# ─────────────────────────────────────────────────────────────────────────────

def run_gwr(coords, X, y_norm, y_raw, y_mean, y_std, te, k, lam=1e-3):
    """
    Adaptive-bandwidth GWR: same global kNN graph as GWF (k neighbours from all
    n points).  Gaussian kernel, WLS solved analytically in original feature
    space with a bias column.
    """
    from sklearn.neighbors import NearestNeighbors as _NNS
    p = X.shape[1]
    n = len(X)
    X_aug = np.hstack([np.ones((n, 1), dtype=np.float32), X])   # (n, p+1)

    nn_obj = _NNS(n_neighbors=k + 1, algorithm='ball_tree',
                  metric='euclidean').fit(coords)
    nbr_dist_all, nbr_idx_all = nn_obj.kneighbors(coords)
    nbr_idx_all  = nbr_idx_all[:, 1:]    # drop self
    nbr_dist_all = nbr_dist_all[:, 1:]

    preds = []
    for qi in te:
        nbr = nbr_idx_all[qi]
        d   = nbr_dist_all[qi]
        h   = d.max() + 1e-8
        w   = np.exp(-0.5 * (d / h) ** 2)

        X_loc = X_aug[nbr]
        y_loc = y_norm[nbr]

        W = np.diag(w)
        A = X_loc.T @ W @ X_loc + lam * np.eye(p + 1)
        b = X_loc.T @ (w * y_loc)
        beta = np.linalg.solve(A, b)
        preds.append(float(X_aug[qi] @ beta))

    pred_raw = np.array(preds, dtype=np.float32) * y_std + y_mean
    return metrics(pred_raw, y_raw[te])


# ─────────────────────────────────────────────────────────────────────────────
# GNNWR — Geographically Neural Network Weighted Regression (Du et al. 2020)
# ─────────────────────────────────────────────────────────────────────────────

class _SWMN(nn.Module):
    """Spatial Weight Matrix Network: normalised distance → positive weight."""
    def __init__(self, hidden=(16, 8)):
        super().__init__()
        dims   = [1] + list(hidden) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        layers.append(nn.Softplus())   # guarantees w > 0
        self.net = nn.Sequential(*layers)

    def forward(self, d_norm):         # (B, k) → (B, k)
        return self.net(d_norm.unsqueeze(-1)).squeeze(-1)


class _GNNWRModel(nn.Module):
    """GNNWR: SWMN-learned weights + WLS in original feature space."""
    def __init__(self, feat_dim, lam=1e-3):
        super().__init__()
        self.swmn = _SWMN()
        self.lam  = lam
        self._p1  = feat_dim + 1

    def forward(self, x_q, x_nbr, y_nbr, dists):
        B, k, p = x_nbr.shape
        h      = dists.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        d_norm = dists / h
        w      = self.swmn(d_norm)                        # (B, k)

        ones_q   = torch.ones(B, 1, device=x_q.device)
        ones_nbr = torch.ones(B, k, 1, device=x_nbr.device)
        xq_aug   = torch.cat([ones_q, x_q], dim=-1)        # (B, p+1)
        xn_aug   = torch.cat([ones_nbr, x_nbr], dim=-1)    # (B, k, p+1)

        w_sqrt = w.sqrt().unsqueeze(-1)
        Xw = xn_aug * w_sqrt
        A  = torch.bmm(Xw.transpose(1, 2), Xw)
        A  = A + self.lam * torch.eye(self._p1, device=A.device).unsqueeze(0)
        b  = torch.bmm(xn_aug.transpose(1, 2),
                       (w * y_nbr).unsqueeze(-1)).squeeze(-1)
        beta  = torch.linalg.solve(A, b)
        return (xq_aug * beta).sum(-1)


def _train_spatial_model(model, train_dl, val_dl, test_dl,
                         forward_fn, epochs, lr, device):
    """Shared training loop for GNNWR and GWR-ANN."""
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    fn    = nn.MSELoss()

    best_val, best_state = float("inf"), None

    print(f"\n{'Epoch':>6}  {'TrainRMSE':>10}  {'ValRMSE':>8}  {'Time':>6}")
    print("─" * 40)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses = []
        for batch in train_dl:
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            yh   = forward_fn(model, batch)
            loss = fn(yh, batch["y"])
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
                val_preds.append(forward_fn(model, batch).cpu())
                val_true.append(batch["y"].cpu())
        val_rmse = math.sqrt(np.mean((torch.cat(val_preds).numpy() -
                                      torch.cat(val_true).numpy()) ** 2))
        if val_rmse < best_val:
            best_val   = val_rmse
            best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            tr_rmse = math.sqrt(np.mean(tr_losses))
            print(f"{ep:>6}  {tr_rmse:>10.4f}  {val_rmse:>8.4f}  "
                  f"{time.time() - t0:>5.1f}s")

    model.load_state_dict(best_state)
    model.eval()
    te_preds, te_true = [], []
    with torch.no_grad():
        for batch in test_dl:
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            te_preds.append(forward_fn(model, batch).cpu())
            te_true.append(batch["y"].cpu())
    return torch.cat(te_preds).numpy(), torch.cat(te_true).numpy()


def run_gnnwr(coords, X, y_norm, y_raw, y_mean, y_std,
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

    model = _GNNWRModel(feat_dim=X.shape[1]).to(device)

    def _fwd(m, b):
        return m(b["x"], b["nbr_x"], b["nbr_y"], b["nbr_dist"])

    pred_norm, true_norm = _train_spatial_model(
        model, train_dl, val_dl, test_dl, _fwd, epochs, lr, device)
    pred_raw = pred_norm * y_std + y_mean
    true_raw = true_norm * y_std + y_mean
    return metrics(pred_raw, true_raw)


# ─────────────────────────────────────────────────────────────────────────────
# GWR-ANN — Geographically Weighted ANN (Hagenauer & Helbich 2017 inspired)
# ─────────────────────────────────────────────────────────────────────────────

class _GWRANNModel(nn.Module):
    """
    Global MLP with geographic context:
      1. Fixed Gaussian spatial weights from distances.
      2. Weighted sum of neighbour features → local context vector.
      3. MLP([x_query, local_context, coord_query]) → y.
    """
    def __init__(self, feat_dim, hidden=(64, 32)):
        super().__init__()
        in_dim = feat_dim + feat_dim + 2    # x_q || x_ctx || coord
        dims   = [in_dim] + list(hidden) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_q, x_nbr, dists, coord):
        h     = dists.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        w     = torch.exp(-0.5 * (dists / h) ** 2)
        w     = w / w.sum(1, keepdim=True)
        x_ctx = (x_nbr * w.unsqueeze(-1)).sum(1)      # (B, p)
        return self.mlp(torch.cat([x_q, x_ctx, coord], dim=-1)).squeeze(-1)


def run_gwrann(coords, X, y_norm, y_raw, y_mean, y_std,
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

    model = _GWRANNModel(feat_dim=X.shape[1]).to(device)

    def _fwd(m, b):
        return m(b["x"], b["nbr_x"], b["nbr_dist"], b["coord"])

    pred_norm, true_norm = _train_spatial_model(
        model, train_dl, val_dl, test_dl, _fwd, epochs, lr, device)
    pred_raw = pred_norm * y_std + y_mean
    true_raw = true_norm * y_std + y_mean
    return metrics(pred_raw, true_raw)


# ─────────────────────────────────────────────────────────────────────────────
# GWF-U  (BNN reparameterization)
# ─────────────────────────────────────────────────────────────────────────────

def run_gwf_u(coords, X, y_norm, y_raw, y_mean, y_std,
              tr, va, te, epochs, lr, k, batch_size, seed, device,
              n_mc=20):

    torch.manual_seed(seed)
    nbr_idx, nbr_dist = build_knn_graph(coords, k)
    dataset = SpatialRegressionDataset(coords, X, y_norm, nbr_idx, nbr_dist)

    train_dl = DataLoader(Subset(dataset, tr), batch_size=batch_size,
                          shuffle=True, drop_last=True, num_workers=0)
    val_dl   = DataLoader(Subset(dataset, va), batch_size=batch_size,
                          shuffle=False, num_workers=0)
    test_dl  = DataLoader(Subset(dataset, te), batch_size=batch_size,
                          shuffle=False, num_workers=0)

    model = GWF_U(feat_dim=X.shape[1]).to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    e_dim = model.z_proj_dim

    best_val   = float("inf")
    best_state = None

    print(f"\n{'Epoch':>6}  {'TrainRMSE':>10}  {'ValRMSE':>8}  {'logσ':>7}  {'Time':>6}")
    print("─" * 48)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses = []
        for batch in train_dl:
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            B   = batch["coord"].shape[0]
            eps = torch.randn(B, e_dim, device=device)
            yh, _ = model(batch, eps_beta=eps)
            loss  = model.loss(yh, batch["y"])
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
                yh, _ = model(batch)          # MAP estimate for val
                val_preds.append(yh.cpu()); val_true.append(batch["y"].cpu())
        val_rmse = math.sqrt(np.mean((torch.cat(val_preds).numpy() -
                                      torch.cat(val_true).numpy()) ** 2))

        if val_rmse < best_val:
            best_val   = val_rmse
            best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            tr_rmse = math.sqrt(np.mean(tr_losses))
            log_s   = model.gwr.log_sigma.item()
            print(f"{ep:>6}  {tr_rmse:>10.4f}  {val_rmse:>8.4f}  "
                  f"{log_s:>7.3f}  {time.time()-t0:>5.1f}s")

    model.load_state_dict(best_state)
    model.eval()

    # MAP point predictions + MC uncertainty (n_mc samples)
    te_preds_map, te_sigma, te_true = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            B = batch["coord"].shape[0]

            yh_map, _ = model(batch)          # MAP
            mc_preds  = []
            for _ in range(n_mc):
                eps  = torch.randn(B, e_dim, device=device)
                yh_i, _ = model(batch, eps_beta=eps)
                mc_preds.append(yh_i)
            sigma = torch.stack(mc_preds).std(dim=0)

            te_preds_map.append(yh_map.cpu())
            te_sigma.append(sigma.cpu())
            te_true.append(batch["y"].cpu())

    pred_norm = torch.cat(te_preds_map).numpy()
    true_norm = torch.cat(te_true).numpy()
    sigma_norm = torch.cat(te_sigma).numpy()

    pred_raw  = pred_norm  * y_std + y_mean
    true_raw  = true_norm  * y_std + y_mean
    sigma_raw = sigma_norm * y_std            # scale σ back to original units

    rmse, r2, mape = metrics(pred_raw, true_raw)
    mean_sigma = float(sigma_raw.mean())
    return rmse, r2, mape, mean_sigma


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
    ap.add_argument("--srgcnn_epochs", type=int,   default=20)
    ap.add_argument("--srgcnn_lr",     type=float, default=3e-2)
    ap.add_argument("--device",        default="cpu")
    args = ap.parse_args()

    print("=" * 62)
    print("  GWF vs SRGCNN-GW vs OLS — San Diego Airbnb  (log_price)")
    print("  Split: 60% train / 20% val / 20% test  (spatial, lat-sorted)")
    print("=" * 62)

    coords, X, y_norm, y_raw, y_mean, y_std = load_airbnb(args.data)
    n = len(X)
    tr, va, te = split_spatial_622(coords)
    print(f"  n={n}  train={len(tr)}  val={len(va)}  test={len(te)}  p={X.shape[1]}")

    results = {}

    # ── OLS ───────────────────────────────────────────────────────────────
    print("\n── OLS ───────────────────────────────────────────────────────")
    results["OLS"] = run_ols(X, y_norm, y_raw, y_mean, y_std, tr, te)
    rmse, r2, mape = results["OLS"]
    print(f"  RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    # ── GWR ───────────────────────────────────────────────────────────────
    print(f"\n── GWR  (analytical, k={args.k}) ─────────────────────────────")
    results["GWR"] = run_gwr(coords, X, y_norm, y_raw, y_mean, y_std,
                             te, k=args.k)
    rmse, r2, mape = results["GWR"]
    print(f"  RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    # ── GNNWR ─────────────────────────────────────────────────────────────
    print(f"\n── GNNWR  ({args.gwf_epochs} epochs, lr={args.gwf_lr}) ──────────────")
    results["GNNWR"] = run_gnnwr(
        coords, X, y_norm, y_raw, y_mean, y_std,
        tr, va, te,
        epochs=args.gwf_epochs, lr=args.gwf_lr,
        k=args.k, batch_size=args.gwf_batch,
        seed=args.seed, device=args.device)
    rmse, r2, mape = results["GNNWR"]
    print(f"\n  Test → RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    # ── GWR-ANN ───────────────────────────────────────────────────────────
    print(f"\n── GWR-ANN  ({args.gwf_epochs} epochs, lr={args.gwf_lr}) ─────────────")
    results["GWR-ANN"] = run_gwrann(
        coords, X, y_norm, y_raw, y_mean, y_std,
        tr, va, te,
        epochs=args.gwf_epochs, lr=args.gwf_lr,
        k=args.k, batch_size=args.gwf_batch,
        seed=args.seed, device=args.device)
    rmse, r2, mape = results["GWR-ANN"]
    print(f"\n  Test → RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    # ── SRGCNN-GW ─────────────────────────────────────────────────────────
    print(f"\n── SRGCNN-GW  ({args.srgcnn_epochs} epochs, lr={args.srgcnn_lr}) ─────────")
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

    # ── GWF-U (BNN) ───────────────────────────────────────────────────────
    print(f"\n── GWF-U  ({args.gwf_epochs} epochs, lr={args.gwf_lr}, n_mc=20) ─")
    gwfu_res = run_gwf_u(
        coords, X, y_norm, y_raw, y_mean, y_std,
        tr, va, te,
        epochs=args.gwf_epochs, lr=args.gwf_lr,
        k=args.k, batch_size=args.gwf_batch,
        seed=args.seed, device=args.device, n_mc=20)
    results["GWF-U"] = gwfu_res[:3]          # (rmse, r2, mape) for table
    rmse, r2, mape, mean_sigma = gwfu_res
    print(f"\n  Test → RMSE={rmse:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%"
          f"  mean_σ={mean_sigma:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FINAL RESULTS — 20% held-out test set")
    print("=" * 62)
    print(f"  {'Model':<18} {'RMSE':>8} {'R²':>8} {'MAPE':>8}")
    print(f"  {'-'*46}")
    for name, (rmse, r2, mape) in results.items():
        print(f"  {name:<18} {rmse:>8.4f} {r2:>8.4f} {mape:>7.2f}%")
    print(f"\n  GWF-U mean predictive σ (log_price scale): {gwfu_res[3]:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
