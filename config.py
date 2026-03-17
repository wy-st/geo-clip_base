# =============================================================================
# GWF 配置文件
# 修改这里的参数，然后运行 python train.py
# =============================================================================

# ── 数据设置 ──────────────────────────────────────────────────────────────────
DATA = {
    # 数据来源: "geojson" | "csv" | "california" | "synthetic"
    "source": "geojson",

    # GeoJSON 文件路径（source="geojson" 时使用）
    "geojson_path": "airbnb_regression_db.geojson",

    # CSV 文件路径（source="csv" 时使用）
    # "csv_path": "data.csv",
    # "lat_col":  "latitude",
    # "lon_col":  "longitude",

    # 预测目标列
    "target_col": "log_price",

    # 特征列列表；None = 自动使用所有数值型列（除了目标列）
    "feature_cols": [
        "accommodates", "bathrooms", "bedrooms", "beds",
        "pool", "d2balboa", "coastal",
        "pg_Apartment", "pg_Condominium", "pg_House", "pg_Townhouse",
        "rt_Entire_home/apt", "rt_Private_room", "rt_Shared_room",
    ],

    # 训练/验证划分比例（0.2 = 20% 用于验证）
    "val_split": 0.2,

    # 随机种子（保证可复现）
    "seed": 42,
}


# ── 模型超参数 ─────────────────────────────────────────────────────────────────
MODEL = {
    # 最近邻数量 k（纯超参数，改这里不需要改模型代码）
    "k_neighbors": 16,

    # GeoCLIP 投影维度（512 → loc_proj_dim）
    "loc_proj_dim": 64,

    # 节点表示维度
    "node_dim": 128,

    # β 和 W_static 的维度（与 k 无关，自由设置）
    "z_proj_dim": 32,

    # 注意力 Q/K 维度
    "attn_dim": 64,

    # y 注入维度（小一点就行，默认 8）
    "y_inject_dim": 8,

    # TabPFN 模型路径；None = 从 HuggingFace 自动下载
    "tabpfn_path": "/root/.cache/tabpfn/tabpfn-v2-regressor.ckpt",
}


# ── 训练超参数 ─────────────────────────────────────────────────────────────────
TRAINING = {
    # 训练轮数
    "epochs": 150,

    # 学习率
    "lr": 3e-4,

    # 权重衰减（L2 正则化）
    "weight_decay": 1e-4,

    # batch 大小
    "batch_size": 256,

    # 设备: "cpu" 或 "cuda"（有 GPU 的话改成 "cuda"）
    "device": "cpu",

    # 模型保存路径
    "save_path": "checkpoints/gwf_best.pt",
}
