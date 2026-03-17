"""
GWF — Geographical Weights Foundation Model
============================================

所有模型组件都在这个文件里。打开这个文件可以看到完整算法。

────────────────────────────────────────────────────────────────────
架构总览 (对每个查询点 i，有 k 个最近邻 j = 1…k)
────────────────────────────────────────────────────────────────────

输入
  coord_i           查询点坐标 (lat, lon)
  x_i               tabular 特征 (p 维)
  coord_j, x_j, y_j 邻居的坐标、特征、标签

STAGE 1 — 位置编码  [冻结，不参与梯度]
  e_i = GeoCLIP(coord_i) → 64 维位置嵌入

STAGE 2 — In-Context 编码  [冻结，不参与梯度]
  z_i, z_j = TabPFN(x_i | x_j, y_j)
  TabPFN 把 k 个邻居当作 in-context 样本，返回 192 维隐状态
  z 比原始特征 x 更丰富，包含了局部数据分布信息

STAGE 3 — 节点投影  [可训练]
  h = GELU( Linear([z ‖ e]) )  →  128 维
  查询点和邻居共用同一个投影层 (shared weights)

STAGE 4 — GWR Context Module  [可训练，核心算法]

  A. y 注入 (把邻居标签信息融入邻居表示)
       y_norm  = (y_j − μ) / σ            ← 局部归一化
       h_aug_j = h_j + W_ya · tanh(W_y · y_norm)

  B. 空间注意力 (同时考虑位置 + 标签分布)
       w = softmax( q(h_i) · k(h_aug)ᵀ / √A − d_norm )

  C. 上下文聚合  (k 无关，k 是超参数)
       c = Σ_j  w_j · h_aug_j

  D. β 生成  (空间变化的局部系数，GWR 核心)
       β_i = Linear( h_i + c )

  E. 预测  (TabPFN 嵌入空间里的双线性点积)
       ŷ_i = (z_i @ W_static) · β_i

输出
  ŷ_i   标量预测值
  β_i   32 维局部系数向量 (可在地图上可视化，体现空间异质性)

────────────────────────────────────────────────────────────────────
可训练参数汇总 (~93K)
  loc_proj     512 → 64     GeoCLIP 投影
  node_proj    256 → 128    节点投影
  y_proj       1   → 8      y 注入
  y_inject     8   → 128
  query_head   128 → 64     注意力 Q
  key_head     128 → 64     注意力 K
  W_static     192 × 32     共享 z 投影
  beta_head    128 → 32     β 生成

冻结的基础模型
  GeoCLIP    预训练位置编码器 (512 维)
  TabPFN     预训练 in-context 回归器 (192 维隐状态)
────────────────────────────────────────────────────────────────────
"""

import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 让 geoclip 包可以被找到 ────────────────────────────────────────────────
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from geoclip.model.location_encoder import LocationEncoder

# TabPFN 加载逻辑比较复杂（需要 hook 内部 API），放在独立文件里
# 功能：把 k 个邻居作为 context，返回 z_query 和 z_nbr（预 MLP 隐状态）
from .tabpfn_encoder import TabPFNInContextEncoder


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — 位置编码（冻结）
# ═════════════════════════════════════════════════════════════════════════════

class GeoCLIPEncoder(nn.Module):
    """
    预训练 GeoCLIP 位置编码器（全部参数冻结）。

    输入 : (N, 2)  [lat, lon] 单位：度
    输出 : (N, 512) L2 归一化位置嵌入
    """

    def __init__(self):
        super().__init__()
        self.encoder = LocationEncoder(from_pretrained=True)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(coords), dim=-1)   # (N, 512)


class LocationFusion(nn.Module):
    """
    GeoCLIP → 可训练线性投影 → loc_proj_dim 维。

    支持 (B, 2) 和 (B, k, 2) 两种输入形状。
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.geo_enc = GeoCLIPEncoder()
        self.proj    = nn.Linear(512, out_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        flat   = coords.reshape(-1, 2)              # 展平为 (N, 2)
        e      = self.geo_enc(flat)                 # (N, 512)
        e_proj = self.proj(e)                       # (N, out_dim)
        return e_proj.reshape(*coords.shape[:-1], -1)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — GWR Context Module（核心可训练算法）
# ═════════════════════════════════════════════════════════════════════════════

class GWRContextModule(nn.Module):
    """
    上下文驱动的 β 生成模块（GWR 核心）。

    步骤：
      A. y 注入     h_aug = h_nbr + W_ya · tanh(W_y · y_norm)
      B. 空间注意力  w     = softmax(q · kᵀ / √A − d_norm)
      C. 上下文聚合  c     = Σ w_j · h_aug_j
      D. β 生成     β     = Linear(h_query + c)
      E. 预测       ŷ     = (z_query @ W_static) · β

    k 是纯超参数，模型无任何维度上的 k 约束。
    """

    def __init__(
        self,
        node_dim:     int,        # H: 节点表示维度
        tabpfn_dim:   int,        # D: TabPFN 嵌入维度
        z_proj_dim:   int = 32,   # E: W_static 和 β 的维度
        attn_dim:     int = 64,   # A: 注意力 Q/K 维度
        y_inject_dim: int = 8,    # y 注入的小维度（8 足够）
    ):
        super().__init__()
        self.attn_dim = attn_dim

        # ── A. y 注入 ───────────────────────────────────────────────────────
        self.y_proj   = nn.Sequential(nn.Linear(1, y_inject_dim), nn.Tanh())
        self.y_inject = nn.Linear(y_inject_dim, node_dim, bias=False)

        # ── B. 空间注意力 ────────────────────────────────────────────────────
        self.query_head = nn.Linear(node_dim, attn_dim, bias=False)
        self.key_head   = nn.Linear(node_dim, attn_dim, bias=False)
        nn.init.xavier_uniform_(self.query_head.weight)
        nn.init.xavier_uniform_(self.key_head.weight)

        # ── D-E. 共享 z 投影 + β 生成头 ─────────────────────────────────────
        self.W_static  = nn.Parameter(torch.zeros(tabpfn_dim, z_proj_dim))
        self.beta_head = nn.Linear(node_dim, z_proj_dim)
        nn.init.normal_(self.W_static, std=0.02)

    def forward(
        self,
        h_query: torch.Tensor,             # (B, H)    查询节点表示
        h_nbr:   torch.Tensor,             # (B, k, H) 邻居节点表示
        y_nbr:   torch.Tensor,             # (B, k)    邻居标签
        z_query: torch.Tensor,             # (B, D)    查询 TabPFN 嵌入
        dist:    torch.Tensor | None,      # (B, k)    地理距离（可选）
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回
        -------
        y_hat : (B,)            预测值
        beta  : (B, z_proj_dim) 局部 GWR 系数（空间变化，可在地图上可视化）
        w     : (B, k)          空间注意力权重
        """
        # ── A. 局部归一化 y，注入到邻居表示 ─────────────────────────────────
        mu     = y_nbr.mean(dim=-1, keepdim=True)
        sigma  = y_nbr.std(dim=-1, keepdim=True).clamp(min=1e-6)
        y_norm = (y_nbr - mu) / sigma                          # (B, k)

        y_feat = self.y_proj(y_norm.unsqueeze(-1))             # (B, k, y_dim)
        h_aug  = h_nbr + self.y_inject(y_feat)                 # (B, k, H)

        # ── B. 注意力分数 ────────────────────────────────────────────────────
        q      = self.query_head(h_query)                      # (B, A)
        k      = self.key_head(h_aug)                          # (B, k, A)
        scores = torch.einsum("ba,bka->bk", q, k) / math.sqrt(self.attn_dim)

        if dist is not None:
            # 距离越远，分数越低（soft 距离衰减）
            d_norm = dist / (dist.max(dim=-1, keepdim=True).values + 1e-8)
            scores = scores - d_norm

        w = F.softmax(scores, dim=-1)                          # (B, k)

        # ── C. 上下文聚合 ─────────────────────────────────────────────────────
        context = torch.einsum("bk,bkh->bh", w, h_aug)        # (B, H)

        # ── D. β 生成 ─────────────────────────────────────────────────────────
        beta = self.beta_head(h_query + context)               # (B, E)

        # ── E. 预测 ───────────────────────────────────────────────────────────
        z_proj = z_query @ self.W_static                       # (B, E)
        y_hat  = (z_proj * beta).sum(dim=-1)                   # (B,)

        return y_hat, beta, w


# ═════════════════════════════════════════════════════════════════════════════
# 完整 GWF 模型
# ═════════════════════════════════════════════════════════════════════════════

class GWF(nn.Module):
    """
    Geographical Weights Foundation Model.

    参数
    ----
    feat_dim      : tabular 特征数量（由数据自动确定）
    loc_proj_dim  : GeoCLIP 投影维度（默认 64）
    node_dim      : 节点表示维度（默认 128）
    z_proj_dim    : β 和 W_static 的维度（默认 32，与 k 无关）
    attn_dim      : 注意力 Q/K 维度（默认 64）
    y_inject_dim  : y 注入维度（默认 8）
    tabpfn_path   : TabPFN .ckpt 路径；None = 从 HuggingFace 下载
    """

    def __init__(
        self,
        feat_dim:     int,
        loc_proj_dim: int        = 64,
        node_dim:     int        = 128,
        z_proj_dim:   int        = 32,
        attn_dim:     int        = 64,
        y_inject_dim: int        = 8,
        tabpfn_path:  str | None = None,
    ):
        super().__init__()
        self.z_proj_dim = z_proj_dim

        # STAGE 1: 位置编码（冻结）
        self.loc_enc = LocationFusion(out_dim=loc_proj_dim)

        # STAGE 2: in-context 编码（冻结）
        self.ctx_enc = TabPFNInContextEncoder(model_path=tabpfn_path)
        tabpfn_dim   = self.ctx_enc.emb_dim      # 通常 192

        # STAGE 3: 节点投影（可训练，查询点和邻居共用）
        self.node_proj = nn.Linear(tabpfn_dim + loc_proj_dim, node_dim)

        # STAGE 4: GWR Context Module（可训练）
        self.ctx_mod = GWRContextModule(
            node_dim     = node_dim,
            tabpfn_dim   = tabpfn_dim,
            z_proj_dim   = z_proj_dim,
            attn_dim     = attn_dim,
            y_inject_dim = y_inject_dim,
        )

    def _project_node(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """[z ‖ e] → GELU → h，节点投影（共用权重）。"""
        return F.gelu(self.node_proj(torch.cat([z, e], dim=-1)))

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数 (batch 字典的键)
        ----------------------
        coord     : (B, 2)
        x         : (B, p)
        nbr_coord : (B, k, 2)
        nbr_x     : (B, k, p)
        nbr_y     : (B, k)
        nbr_dist  : (B, k)

        返回
        ----
        y_hat : (B,)            预测值
        beta  : (B, z_proj_dim) 局部 GWR 系数
        """
        coord    = batch["coord"]
        x        = batch["x"]
        nbr_coord = batch["nbr_coord"]
        nbr_x    = batch["nbr_x"]
        nbr_y    = batch["nbr_y"]
        nbr_dist = batch["nbr_dist"]

        B, k, p = nbr_x.shape

        # STAGE 2: TabPFN in-context 编码
        # 如果 batch 里已有预计算嵌入就直接用（train.py 会提前缓存），否则实时计算
        if "z_query" in batch:
            z_query = batch["z_query"]              # (B, D)
            z_nbr   = batch["z_nbr"]                # (B, k, D)
        else:
            z_query, z_nbr = self.ctx_enc(x, nbr_x, nbr_y)    # (B,D), (B,k,D)

        # STAGE 1: GeoCLIP 位置编码
        e_query = self.loc_enc(coord)                           # (B, L)
        e_nbr   = self.loc_enc(nbr_coord)                       # (B, k, L)

        # STAGE 3: 节点投影（展平后投影，再 reshape 回来）
        h_query = self._project_node(z_query, e_query)          # (B, H)
        h_nbr   = self._project_node(
            z_nbr.reshape(B * k, -1),
            e_nbr.reshape(B * k, -1),
        ).reshape(B, k, -1)                                     # (B, k, H)

        # STAGE 4: GWR Context Module → 预测 + β
        y_hat, beta, _ = self.ctx_mod(
            h_query, h_nbr, nbr_y, z_query, dist=nbr_dist)

        return y_hat, beta

    @torch.no_grad()
    def predict(self, dataloader, device: str = "cpu"):
        """
        对整个 DataLoader 做推理。

        返回
        ----
        y_hat  : (N,)
        beta   : (N, z_proj_dim)  — 局部 GWR 系数，可在地图上可视化
        coords : (N, 2)
        y_true : (N,)
        """
        self.eval()
        preds, betas, coords_list, ys = [], [], [], []
        for batch in dataloader:
            batch = {kk: vv.to(device) for kk, vv in batch.items()
                     if isinstance(vv, torch.Tensor)}
            yh, beta = self(batch)
            preds.append(yh.cpu())
            betas.append(beta.cpu())
            coords_list.append(batch["coord"].cpu())
            ys.append(batch["y"].cpu())
        return (torch.cat(preds),
                torch.cat(betas),
                torch.cat(coords_list),
                torch.cat(ys))
