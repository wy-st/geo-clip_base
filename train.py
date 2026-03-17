"""
GWF 训练脚本

使用方法：
    python train.py

所有参数在 config.py 里修改，不需要动这个文件。
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 确保项目根目录在 Python 路径里
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA, MODEL, TRAINING
from gwf.model import GWF
from gwf.data  import get_dataloaders


# ─────────────────────────────────────────────────────────────────────────────

def precompute_embeddings(model, full_dataset, device, batch_size=256):
    """
    TabPFN 参数全部冻结，所以对固定的 k-NN 图，每个点的嵌入是常数。
    训练前一次性算好，之后每个 epoch 直接查表，速度提升 10-50x。

    full_dataset : SpatialRegressionDataset（random_split 之前的原始数据集）
    """
    from torch.utils.data import DataLoader as _DL

    n = len(full_dataset)
    D = model.ctx_enc.emb_dim
    k = int(full_dataset.nbr_idx.shape[1])

    z_query_all = torch.zeros(n, D,    dtype=torch.float32)
    z_nbr_all   = torch.zeros(n, k, D, dtype=torch.float32)

    loader = _DL(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.ctx_enc.eval()

    offset = 0
    n_batches = len(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            z_q, z_n = model.ctx_enc(batch["x"], batch["nbr_x"], batch["nbr_y"])
            idxs = batch["idx"]
            z_query_all[idxs] = z_q.cpu().float()
            z_nbr_all[idxs]   = z_n.cpu().float()
            offset += z_q.shape[0]
            print(f"\r  [{i+1}/{n_batches}] {offset}/{n} 点", end="", flush=True)

    print(f"\r  完成，缓存 {n:,} 条嵌入 "
          f"({(z_query_all.nbytes + z_nbr_all.nbytes) / 1e6:.0f} MB)  ")

    full_dataset.z_query = z_query_all
    full_dataset.z_nbr   = z_nbr_all


def evaluate(model, dataloader, device):
    """在 dataloader 上计算 RMSE 和 R²。"""
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            y_hat, _ = model(batch)
            y_preds.append(y_hat.cpu())
            y_trues.append(batch["y"].cpu())

    y_pred = torch.cat(y_preds)
    y_true = torch.cat(y_trues)

    rmse = float(F.mse_loss(y_pred, y_true).sqrt())
    ss_res = float(((y_pred - y_true) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    return rmse, r2


# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = TRAINING["device"]
    seed   = DATA.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n" + "=" * 60)
    print("  GWF — Geographical Weights Foundation Model")
    print("=" * 60)

    # ── 1. 加载数据 ────────────────────────────────────────────────────────
    print("\n[1/4] 加载数据...")
    train_dl, val_dl, feat_dim = get_dataloaders(
        source       = DATA["source"],
        k            = MODEL["k_neighbors"],
        val_split    = DATA.get("val_split", 0.2),
        batch_size   = TRAINING["batch_size"],
        seed         = seed,
        geojson_path = DATA.get("geojson_path"),
        csv_path     = DATA.get("csv_path"),
        lat_col      = DATA.get("lat_col", "lat"),
        lon_col      = DATA.get("lon_col", "lon"),
        target_col   = DATA.get("target_col", "price"),
        feature_cols = DATA.get("feature_cols"),
    )
    print(f"  训练集: {len(train_dl.dataset):,} 条  "
          f"验证集: {len(val_dl.dataset):,} 条  "
          f"特征数: {feat_dim}")

    # ── 2. 构建模型 ────────────────────────────────────────────────────────
    print("\n[2/4] 构建模型...")
    model = GWF(
        feat_dim     = feat_dim,
        loc_proj_dim = MODEL.get("loc_proj_dim", 64),
        node_dim     = MODEL.get("node_dim", 128),
        z_proj_dim   = MODEL.get("z_proj_dim", 32),
        attn_dim     = MODEL.get("attn_dim", 64),
        y_inject_dim = MODEL.get("y_inject_dim", 8),
        tabpfn_path  = MODEL.get("tabpfn_path"),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {n_params:,}")

    # ── 3a. 预计算 TabPFN 嵌入（只跑一次，之后每 epoch 直接查表）──────────
    print("\n[3/4] 预计算 TabPFN 嵌入（只需等一次）...")
    # train_dl.dataset 是 Subset，.dataset 才是原始 SpatialRegressionDataset
    full_dataset = train_dl.dataset.dataset
    precompute_embeddings(model, full_dataset, device,
                          batch_size=TRAINING["batch_size"])

    # ── 3b. 训练 ────────────────────────────────────────────────────────────
    print("\n训练中...")
    epochs = TRAINING.get("epochs", 150)
    opt    = AdamW(
        model.parameters(),
        lr           = TRAINING.get("lr", 3e-4),
        weight_decay = TRAINING.get("weight_decay", 1e-4),
    )
    sched = CosineAnnealingLR(opt, T_max=epochs,
                               eta_min=TRAINING.get("lr", 3e-4) * 0.05)

    save_path = TRAINING.get("save_path", "checkpoints/gwf_best.pt")
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    best_r2   = -float("inf")
    best_rmse = float("inf")

    print(f"  {'Epoch':>5}  {'train_loss':>10}  {'val_RMSE':>9}  {'val_R²':>7}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*9}  {'-'*7}")

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            y_hat, _ = model(batch)
            loss = F.mse_loss(y_hat, batch["y"])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())

        sched.step()

        # 每 10 轮或最后一轮打印
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            val_rmse, val_r2 = evaluate(model, val_dl, device)
            train_loss = float(np.mean(losses))
            marker = " ◀ best" if val_r2 > best_r2 else ""
            print(f"  {epoch:5d}  {train_loss:10.4f}  {val_rmse:9.4f}  {val_r2:7.4f}{marker}")

            if val_r2 > best_r2:
                best_r2   = val_r2
                best_rmse = val_rmse
                torch.save(model.state_dict(), save_path)

    # ── 4. 最终结果 ────────────────────────────────────────────────────────
    print("\n[4/4] 最终评估...")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    val_rmse, val_r2 = evaluate(model, val_dl, device)

    print(f"\n{'─' * 40}")
    print(f"  验证集 RMSE : {val_rmse:.4f}")
    print(f"  验证集 R²   : {val_r2:.4f}")
    print(f"  模型已保存  : {save_path}")
    print(f"{'─' * 40}\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
