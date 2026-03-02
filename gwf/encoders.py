"""
GWF — Location Encoder

GeoCLIPEncoder : wraps the pretrained GeoCLIP location encoder (frozen).
LocationFusion : thin linear projection on top of GeoCLIP output.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Make geoclip importable from the parent package ───────────────────────────
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from geoclip.model.location_encoder import LocationEncoder


# ──────────────────────────────────────────────────────────────────────────────
# GeoCLIP Location Encoder (pretrained, frozen)
# ──────────────────────────────────────────────────────────────────────────────

class GeoCLIPEncoder(nn.Module):
    """
    Wraps GeoCLIP's pretrained LocationEncoder.
    All parameters are frozen — used purely as a feature extractor.

    Input  : (B, 2)  [lat, lon] in degrees
    Output : (B, 512) L2-normalised location embedding
    """
    def __init__(self):
        super().__init__()
        self.encoder = LocationEncoder(from_pretrained=True)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(coords)                   # (B, 512)
        return F.normalize(emb, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Location Fusion: GeoCLIP → linear projection
# ──────────────────────────────────────────────────────────────────────────────

class LocationFusion(nn.Module):
    """
    Project frozen GeoCLIP embeddings into a compact location representation.

    Input  : coords (B, 2) or (B, k, 2)
    Output : e_loc  (B, out_dim) or (B, k, out_dim)
    """
    def __init__(self, geo_dim: int = 512, out_dim: int = 64):
        super().__init__()
        self.geo_enc = GeoCLIPEncoder()
        self.proj    = nn.Linear(geo_dim, out_dim, bias=True)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        flat  = coords.reshape(-1, 2)
        e_geo = self.geo_enc(flat)                   # (*, 512)
        e_loc = self.proj(e_geo)                     # (*, out_dim)
        return e_loc.reshape(*coords.shape[:-1], -1)
