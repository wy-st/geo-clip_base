"""
GWF — Location Encoders

GeoCLIPEncoder  : wraps the pretrained GeoCLIP location encoder (frozen).
SatCLIPEncoder  : a SatCLIP-compatible location encoder.
                  • If the real `satclip` package is installed, loads its weights.
                  • Otherwise falls back to an RFF-based encoder with satellite-
                    scale sigma values (2**2, 2**6, 2**10) — different from
                    GeoCLIP's (2**0, 2**4, 2**8) so the two encoders are
                    complementary in scale.
LocationFusion  : fuses geo + sat embeddings with a single linear layer.
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

from geoclip.model.location_encoder import LocationEncoder, LocationEncoderCapsule
from geoclip.model.location_encoder import equal_earth_projection


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
# SatCLIP Proxy — satellite-scale RFF location encoder
# ──────────────────────────────────────────────────────────────────────────────

class _SatCLIPProxy(nn.Module):
    """
    RFF-based location encoder with satellite-scale sigma values.

    GeoCLIP uses σ ∈ {1, 16, 256} — fine-to-coarse global scales.
    SatCLIP operates on overhead imagery with higher spatial resolution,
    so we mirror this with σ ∈ {4, 64, 1024} — shifted by ~2 octaves,
    giving complementary frequency content.

    Architecture per capsule is identical to GeoCLIP's LocationEncoderCapsule
    so we can reuse that class directly.

    Input  : (B, 2)  [lat, lon] in degrees
    Output : (B, 512) L2-normalised embedding
    """
    _SIGMA = [2**2, 2**6, 2**10]   # 4, 64, 1024

    def __init__(self):
        super().__init__()
        self.sigma = self._SIGMA
        for i, s in enumerate(self.sigma):
            self.add_module(f"LocEnc{i}", LocationEncoderCapsule(sigma=s))

    def forward(self, location: torch.Tensor) -> torch.Tensor:
        location = equal_earth_projection(location)
        out = torch.zeros(location.shape[0], 512, device=location.device,
                          dtype=location.dtype)
        for i in range(len(self.sigma)):
            out = out + self._modules[f"LocEnc{i}"](location)
        return F.normalize(out, dim=-1)


class SatCLIPEncoder(nn.Module):
    """
    SatCLIP location encoder.
    Tries to load the official `satclip` package first;
    falls back to _SatCLIPProxy if not available.

    Input  : (B, 2)  [lat, lon] in degrees
    Output : (B, 512) L2-normalised embedding
    """
    def __init__(self, freeze: bool = True):
        super().__init__()
        self._loaded_official = False
        try:
            import satclip                                           # type: ignore
            self.encoder = satclip.load()
            self._loaded_official = True
            print("[SatCLIPEncoder] loaded official SatCLIP weights ✓")
        except Exception:
            print("[SatCLIPEncoder] satclip not available → using RFF proxy")
            self.encoder = _SatCLIPProxy()

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self._loaded_official:
            with torch.no_grad():
                return F.normalize(self.encoder.encode_location(coords), dim=-1)
        else:
            with torch.no_grad():
                return self.encoder(coords)


# ──────────────────────────────────────────────────────────────────────────────
# Fusion: geo ⊕ sat  →  single projected embedding
# ──────────────────────────────────────────────────────────────────────────────

class LocationFusion(nn.Module):
    """
    Fuse GeoCLIP and SatCLIP embeddings with a single linear projection.

    Learnable mixing weight α (sigmoid-activated) controls the geo/sat balance;
    the concatenated [geo; sat] → Linear → loc_proj_dim.

    Input  : coords (B, 2)
    Output : e_loc  (B, loc_proj_dim)
    """
    def __init__(self, geo_dim: int = 512, sat_dim: int = 512,
                 out_dim: int = 256):
        super().__init__()
        self.geo_enc = GeoCLIPEncoder()
        self.sat_enc = SatCLIPEncoder(freeze=True)

        # Single linear layer — minimal MLP philosophy
        self.proj = nn.Linear(geo_dim + sat_dim, out_dim, bias=True)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords : (B, 2) or (B, k, 2) — handles nested neighbourhood tensors
        """
        flat = coords.reshape(-1, 2)
        e_geo = self.geo_enc(flat)          # (*B, 512) — no grad
        e_sat = self.sat_enc(flat)          # (*B, 512) — no grad
        e_loc = self.proj(torch.cat([e_geo, e_sat], dim=-1))  # (*B, out_dim)
        return e_loc.reshape(*coords.shape[:-1], -1)
