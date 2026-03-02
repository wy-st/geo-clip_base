"""
Lightweight SatCLIP location-encoder loader.

Loads only the location encoder from a SatCLIP checkpoint,
without importing the full training pipeline (lightning, etc.).

Usage
-----
    from gwf.satclip_src.load_satclip import load_satclip_loc_encoder

    enc = load_satclip_loc_encoder("satclip-resnet50-l10.ckpt", device="cpu")
    enc.eval()
    with torch.no_grad():
        emb = enc(coords.double())   # (B, embed_dim)

Pretrained checkpoints (download manually from HuggingFace):
    microsoft/SatCLIP-ResNet50-L10   satclip-resnet50-l10.ckpt   ~50 MB   (lightest)
    microsoft/SatCLIP-ResNet50-L40   satclip-resnet50-l40.ckpt
    microsoft/SatCLIP-ViT16-L10      satclip-vit16-l10.ckpt
    microsoft/SatCLIP-ViT16-L40      satclip-vit16-l40.ckpt
"""

import os
import sys
import torch

# ── Make the satclip_src package importable without a proper install ──────────
_here = os.path.dirname(os.path.abspath(__file__))
_pe   = os.path.join(_here, "positional_encoding")
for _p in [_here, _pe]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from location_encoder import (      # noqa: E402  (after sys.path manipulation)
    get_neural_network,
    get_positional_encoding,
    LocationEncoder,
)


def load_satclip_loc_encoder(ckpt_path: str, device: str = "cpu") -> LocationEncoder:
    """
    Load only the location encoder from a SatCLIP checkpoint.

    Parameters
    ----------
    ckpt_path : str   path to the .ckpt file
    device    : str   "cpu" or "cuda"

    Returns
    -------
    loc_encoder : LocationEncoder (double precision, eval mode)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    hp   = ckpt["hyper_parameters"]

    posenc = get_positional_encoding(
        hp["le_type"],
        hp.get("legendre_polys",       10),
        hp.get("harmonics_calculation", "analytic"),
        hp.get("min_radius",             1),
        hp.get("max_radius",           360),
        hp.get("frequency_num",         10),
    )

    nnet = get_neural_network(
        hp["pe_type"],
        posenc.embedding_dim,
        hp.get("embed_dim",     256),
        hp.get("capacity",      256),
        hp.get("num_hidden_layers", 2),
    )

    # Filter only nnet params from the state dict
    state_dict = ckpt["state_dict"]
    state_dict = {
        k[k.index("nnet"):]: v
        for k, v in state_dict.items()
        if "nnet" in k
    }

    loc_encoder = LocationEncoder(posenc, nnet).double()
    loc_encoder.load_state_dict(state_dict)
    loc_encoder.eval()
    loc_encoder.to(device)
    return loc_encoder
