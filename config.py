"""
config.py
=========
All hyperparameters for GWF in one place.
Edit this file to change any setting — no CLI arguments needed.
Then run:  python train.py

Sections:
  DATA       : dataset source, columns, split ratio
  ENCODERS   : paths to pretrained foundation model checkpoints
  MODEL      : architecture hyperparameters
  LOSS       : loss function weights
  TRAINING   : epochs, learning rates, batch size, device
  PATHS      : checkpoint save paths
"""

# =============================================================================
# DATA
# =============================================================================
DATA = {
    # ---- Source ----
    # "geojson" → load from GeoJSON FeatureCollection file
    # "csv"     → load from CSV with lat/lon columns
    "source":       "geojson",
    "path":         "airbnb_regression_db.geojson",   # path to data file

    # ---- Target and features ----
    "target_col":   "log_price",
    "feature_cols": [
        "accommodates", "bathrooms", "bedrooms", "beds", "pool",
        "d2balboa", "coastal",
        "pg_Apartment", "pg_Condominium", "pg_House", "pg_Townhouse",
        "rt_Entire_home/apt", "rt_Private_room", "rt_Shared_room",
    ],

    # ---- For CSV source only ----
    "lat_col":  "lat",
    "lon_col":  "lon",

    # ---- Train/val split ----
    "val_split":  0.2,   # fraction of data used for validation
    "seed":       42,
    "normalise":  True,  # z-normalise X_tab features

    # ---- Prompt for LLM channel ----
    # Either use a key from gwf/data/prompts.py PROMPT_TEMPLATES,
    # or write a custom string in "prompt_text".
    "prompt_key":  "airbnb_sandiego",  # set to "" to use custom prompt_text
    "prompt_text": "",                 # used only when prompt_key == ""
}

# =============================================================================
# ENCODERS  (paths to pretrained foundation model checkpoints)
# =============================================================================
ENCODERS = {
    # ---- SatCLIP: satellite-level location encoder (REQUIRED) ----
    # Download: https://github.com/microsoft/satclip
    "satclip_ckpt":  "",   # e.g. "/data/checkpoints/satclip-resnet18-l10.ckpt"

    # ---- GeoCLIP: street-view location encoder (REQUIRED) ----
    # Weights are bundled in geoclip/model/weights/ — leave empty for auto-load
    "geoclip_ckpt":  "",

    # ---- SkySense++: visual RS encoder (OPTIONAL) ----
    "skysense_ckpt": "",   # leave empty → zero embeddings (fusion learns to ignore)

    # ---- AnyGraph: graph encoder (OPTIONAL) ----
    "anygraph_ckpt": "",   # leave empty → zero embeddings

    # ---- TabPFN: tabular encoder (REQUIRED) ----
    "tabpfn_ckpt":   "/root/.cache/tabpfn/tabpfn-v2-regressor.ckpt",

    # ---- LLM: world knowledge encoder (REQUIRED) ----
    # HuggingFace model name or local path.
    # Recommended: "Qwen/Qwen2.5-7B" or "meta-llama/Llama-3.1-8B-Instruct"
    "llm_name":      "Qwen/Qwen2.5-7B",

    # ---- Device for loading frozen encoders ----
    "device":        "cuda",   # "cuda" or "cpu"
}

# =============================================================================
# MODEL  (architecture hyperparameters)
# =============================================================================
MODEL = {
    "d":              512,    # unified embedding dimension
    "r":               32,    # Beta low-rank dimension
    "k_neighbors":     15,    # kNN spatial graph connectivity
    "num_gnn_layers":   3,    # number of SpatialGNN layers
    "probabilistic":  False,  # False → V1 deterministic | True → V2 variational
    "num_targets":      1,    # 1 for regression, num_classes for classification
    "task":        "regression",   # "regression" or "classification"
}

# =============================================================================
# LOSS
# =============================================================================
LOSS = {
    "lambda_smooth":           0.1,   # weight for spatial smoothness loss
    "smooth_sigma_weight":     1.0,   # sigma term weight within L_smooth
    "smooth_u_weight":         0.1,   # U term weight within L_smooth
    "lambda_kl":              0.01,   # KL divergence weight (V2 only)
    "kl_annealing_epochs":     20,    # KL ramp-up duration in epochs (V2 only)
    "lambda_contrast":         0.1,   # contrastive loss weight (Phase 1 only)
    "contrastive_temperature": 0.1,   # InfoNCE temperature (Phase 1 only)
}

# =============================================================================
# TRAINING
# =============================================================================
TRAINING = {
    "device":       "cuda",   # "cuda" or "cpu"

    # Batch size: for small datasets (N<10k) try full-batch (set to N).
    # For large datasets use 512-1024.
    "batch_size":   512,

    # ---- Phase 1: Bridge warm-up ----
    # Trains ONLY MLP bridges + CrossChannelFusion to align encoder channels.
    "phase1_epochs":  5,
    "phase1_lr":      1e-3,

    # ---- Phase 2: Joint training (main phase) ----
    # Trains all trainable modules end-to-end.
    "phase2_epochs": 100,
    "phase2_lr":     5e-4,

    # ---- Phase 3: Task fine-tuning (optional) ----
    # Trains only FiLM + OutputHead. Use for transfer learning.
    "phase3_epochs":  10,
    "phase3_lr":      1e-4,

    # ---- Shared ----
    "weight_decay":   0.01,
    "grad_clip":      1.0,    # max gradient norm for clipping
    "num_workers":    0,      # DataLoader workers (0 = main process)
}

# =============================================================================
# PATHS
# =============================================================================
PATHS = {
    "checkpoint_dir":  "checkpoints",
    "best_model_path": "checkpoints/gwf_best.pt",
    "log_dir":         "logs",
}
