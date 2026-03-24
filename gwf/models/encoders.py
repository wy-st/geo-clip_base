"""
gwf/models/encoders.py
======================
Module 1: Frozen Encoder Bank

All 6 encoder channels are loaded here, set to eval() mode, and wrapped in
torch.no_grad() during forward passes. No gradients flow through these modules.

Channels:
  1. SatCLIP   (coords → 512)   satellite-level location semantics
  2. GeoCLIP   (coords → 512)   street-view-level location semantics
  3. SkySense++ (image  → 768)  visual RS features              [OPTIONAL]
  4. AnyGraph  (subgraph→ 256)  graph structural features       [OPTIONAL]
  5. TabPFN    (X_tab  → 512)   deep tabular representations
  6. LLM       (text   → 1024)  world knowledge / task context  [OPTIONAL]

For missing modalities the encoder returns a zero tensor of the correct shape
and logs a one-time warning. The fusion layer learns to ignore zero channels
via its attention mechanism.
"""

import sys
import os
import math
import warnings
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: freeze all parameters of a module
# ---------------------------------------------------------------------------

def freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module.eval()


# =============================================================================
# Channel 1: SatCLIP — satellite-level location encoder
# =============================================================================

class SatCLIPEncoder(nn.Module):
    """
    Wraps the SatCLIP location encoder.
    Input : coords (N, 2)  — (latitude, longitude)
    Output: z      (N, 512)
    """

    OUT_DIM = 512

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        super().__init__()
        self.device_str = device
        self._available = False

        try:
            # Use the local satclip_src included in this repo
            repo_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(repo_root))
            from gwf.satclip_src.load import get_satclip   # type: ignore

            if ckpt_path and Path(ckpt_path).exists():
                model = get_satclip(ckpt_path, device=device)
            else:
                # Try default cache location
                default = Path.home() / ".cache" / "satclip" / "satclip-resnet18-l10.ckpt"
                if default.exists():
                    model = get_satclip(str(default), device=device)
                else:
                    raise FileNotFoundError("SatCLIP checkpoint not found.")

            self.loc_enc = freeze(model.location_encoder)
            self._available = True
            logger.info("SatCLIP loaded ✓")
        except Exception as e:
            logger.warning(f"SatCLIP unavailable ({e}). Using zero embeddings.")
            self.loc_enc = None

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 2) lat/lon → (N, 512)"""
        N = coords.shape[0]
        if not self._available:
            return torch.zeros(N, self.OUT_DIM, device=coords.device)

        # SatCLIP location encoder expects (lat, lon) on the same device
        z = self.loc_enc(coords)          # (N, 512)
        z = F.normalize(z, dim=-1)
        return z


# =============================================================================
# Channel 2: GeoCLIP — street-view-level location encoder
# =============================================================================

class GeoCLIPEncoder(nn.Module):
    """
    Wraps the GeoCLIP location encoder (already part of this repo).
    Input : coords (N, 2)  — (latitude, longitude)
    Output: z      (N, 512)
    """

    OUT_DIM = 512

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device_str = device
        self._available = False

        try:
            repo_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(repo_root))
            from geoclip.model.location_encoder import LocationEncoder   # type: ignore

            self.loc_enc = freeze(LocationEncoder())
            # Load pretrained weights
            weight_path = repo_root / "geoclip" / "model" / "weights" / "location_encoder_weights.pth"
            if weight_path.exists():
                state = torch.load(str(weight_path), map_location=device)
                self.loc_enc.load_state_dict(state, strict=False)
                logger.info("GeoCLIP loaded ✓")
            else:
                logger.warning("GeoCLIP weights not found, using random init.")
            self._available = True
        except Exception as e:
            logger.warning(f"GeoCLIP unavailable ({e}). Using zero embeddings.")
            self.loc_enc = None

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 2) lat/lon → (N, 512)"""
        N = coords.shape[0]
        if not self._available:
            return torch.zeros(N, self.OUT_DIM, device=coords.device)

        z = self.loc_enc(coords)          # (N, 512)
        z = F.normalize(z, dim=-1)
        return z


# =============================================================================
# Channel 3: SkySense++ — visual RS feature extractor  [OPTIONAL]
# =============================================================================

class SkySensePPEncoder(nn.Module):
    """
    Wraps SkySense++ backbone (Swin-L based).
    Input : images (N, C, H, W)
    Output: z      (N, 768)

    If the checkpoint is not available, returns zeros with a warning.
    To enable: provide the SkySense++ checkpoint path in config.
    """

    OUT_DIM = 768

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        super().__init__()
        self._available = False

        if ckpt_path and Path(ckpt_path).exists():
            try:
                # SkySense++ loading requires its own package.
                # Example (adapt based on actual API):
                #   from skysense import build_model
                #   model = build_model(ckpt_path)
                #   self.backbone = freeze(model.backbone)
                raise NotImplementedError(
                    "SkySense++ loading not yet implemented. "
                    "Install the skysense package and add loading code here."
                )
            except Exception as e:
                logger.warning(f"SkySense++ unavailable ({e}).")
        else:
            logger.info("SkySense++ checkpoint not provided. Using zero embeddings.")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (N, C, H, W) → (N, 768)"""
        N = images.shape[0]
        if not self._available:
            return torch.zeros(N, self.OUT_DIM, device=images.device)

        # When implemented:
        #   feats = self.backbone(images)  # (N, 768, H', W')
        #   z = feats.mean(dim=[-2, -1])   # global average pool → (N, 768)
        #   return F.normalize(z, dim=-1)
        return torch.zeros(N, self.OUT_DIM, device=images.device)


# =============================================================================
# Channel 4: AnyGraph — graph structural encoder  [OPTIONAL]
# =============================================================================

class AnyGraphEncoder(nn.Module):
    """
    Wraps AnyGraph (KDD 2025). Encodes local POI/road subgraphs.
    Input : list of PyG Data objects (N subgraphs)
    Output: z (N, 256) graph-level embeddings

    If the checkpoint is not available, returns zeros.
    """

    OUT_DIM = 256

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        super().__init__()
        self._available = False

        if ckpt_path and Path(ckpt_path).exists():
            try:
                # AnyGraph loading (adapt based on actual API):
                #   from anygraph import load_anygraph
                #   self.gnn = freeze(load_anygraph(ckpt_path))
                raise NotImplementedError(
                    "AnyGraph loading not yet implemented. "
                    "Install the anygraph package and add loading code here."
                )
            except Exception as e:
                logger.warning(f"AnyGraph unavailable ({e}).")
        else:
            logger.info("AnyGraph checkpoint not provided. Using zero embeddings.")

    @torch.no_grad()
    def forward(self, subgraphs, N: int, device: torch.device) -> torch.Tensor:
        """subgraphs: list[PyG Data] → (N, 256)"""
        if not self._available:
            return torch.zeros(N, self.OUT_DIM, device=device)

        # When implemented:
        #   from torch_geometric.data import Batch
        #   batch = Batch.from_data_list(subgraphs).to(device)
        #   z = self.gnn(batch)   # (N, 256) graph-level
        #   return F.normalize(z, dim=-1)
        return torch.zeros(N, self.OUT_DIM, device=device)


# =============================================================================
# Channel 5: TabPFN — deep tabular encoder
# =============================================================================

class TabPFNEncoder(nn.Module):
    """
    Uses the pretrained TabPFN to extract deep tabular representations.
    We extract features BEFORE TabPFN's final output head — the internal
    representation captures non-linear feature interactions.

    Input : X_tab (N, p)  — raw tabular features (p varies per dataset)
            y     (N,)    — training labels (only used to set context)
    Output: z     (N, 512)

    Note: TabPFN is an in-context learner. We use it in "feature extraction"
    mode by providing the full training set as context, then querying each point.
    For large datasets, we split into chunks.
    """

    OUT_DIM = 512

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        super().__init__()
        self.device_str = device
        self._available = False
        self._warned = False

        try:
            # Try to load using the local tabpfn_encoder module
            repo_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(repo_root))
            from gwf.tabpfn_encoder import TabPFNEncoder as _TabPFNEncoderImpl  # type: ignore

            self._impl = _TabPFNEncoderImpl(device=device)
            freeze(self._impl)
            self._available = True
            logger.info("TabPFN loaded ✓")
        except Exception as e:
            logger.warning(f"TabPFN unavailable ({e}). Using zero embeddings.")
            self._impl = None

    @torch.no_grad()
    def forward(self, X_tab: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        X_tab: (N, p) raw tabular features
        y    : (N,)   labels (used as in-context targets, optional)
        → (N, 512)
        """
        N = X_tab.shape[0]
        if not self._available:
            return torch.zeros(N, self.OUT_DIM, device=X_tab.device)

        try:
            z = self._impl(X_tab, y)   # (N, 512)
            return z
        except Exception as e:
            if not self._warned:
                logger.warning(f"TabPFN forward failed ({e}). Using zeros.")
                self._warned = True
            return torch.zeros(N, self.OUT_DIM, device=X_tab.device)


# =============================================================================
# Channel 6: LLM — world knowledge encoder  [OPTIONAL but recommended]
# =============================================================================

class LLMEncoder(nn.Module):
    """
    Encodes the dataset/task description prompt using a frozen LLM.
    Returns a single vector (1, 1024) that is broadcast to (N, 1024).

    Supported models:
      - "Qwen/Qwen2.5-7B"   (recommended)
      - "meta-llama/Llama-3.1-8B-Instruct"
      - Any HuggingFace text model

    The prompt is encoded ONCE per dataset and cached. Subsequent calls
    with the same prompt_text reuse the cached vector.

    If transformers is not installed or the model is not available,
    returns zeros with a warning.
    """

    OUT_DIM = 1024

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B", device: str = "cpu"):
        super().__init__()
        self.device_str = device
        self._available = False
        self._cache: dict[str, torch.Tensor] = {}

        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore

            logger.info(f"Loading LLM: {model_name} (this may take a while)...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.lm = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(device)
            freeze(self.lm)

            # Get actual hidden dim; project to OUT_DIM if needed
            hidden_dim = self.lm.config.hidden_size
            if hidden_dim != self.OUT_DIM:
                # This is a trainable projection — placed here for convenience
                self.proj = nn.Linear(hidden_dim, self.OUT_DIM, bias=False)
            else:
                self.proj = nn.Identity()

            self._available = True
            logger.info(f"LLM ({model_name}) loaded ✓")
        except Exception as e:
            logger.warning(f"LLM unavailable ({e}). Using zero task embeddings.")
            self.tokenizer = None
            self.lm = None
            self.proj = None

    @torch.no_grad()
    def encode_prompt(self, prompt_text: str, device: torch.device) -> torch.Tensor:
        """
        Encode prompt_text → (1, OUT_DIM).
        Result is cached by prompt string.
        """
        if prompt_text in self._cache:
            return self._cache[prompt_text].to(device)

        if not self._available:
            result = torch.zeros(1, self.OUT_DIM, device=device)
            self._cache[prompt_text] = result
            return result

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device_str)

        outputs = self.lm(**inputs, output_hidden_states=False)
        # Mean pool over token dimension
        token_embs = outputs.last_hidden_state  # (1, T, H)
        c = token_embs.mean(dim=1)              # (1, H)
        c = self.proj(c.float())                # (1, OUT_DIM)
        c = F.normalize(c, dim=-1)

        result = c.to(device).detach()
        self._cache[prompt_text] = result
        return result

    def forward(self, prompt_text: str, N: int, device: torch.device) -> torch.Tensor:
        """
        Returns (N, OUT_DIM) — prompt embedding broadcast to all N points.
        """
        c = self.encode_prompt(prompt_text, device)  # (1, OUT_DIM)
        return c.expand(N, -1)                        # (N, OUT_DIM)


# =============================================================================
# FrozenEncoderBank — assembles all 6 channels
# =============================================================================

class FrozenEncoderBank(nn.Module):
    """
    Assembles all 6 frozen encoder channels.

    Usage:
        bank = FrozenEncoderBank(cfg)
        z_dict = bank(
            coords=coords,        # (N, 2)  required
            X_tab=X_tab,          # (N, p)  required
            y=y_train,            # (N,)    optional, used by TabPFN as context
            images=images,        # (N,C,H,W) optional
            subgraphs=subgraphs,  # list[PyG Data] optional
            prompt_text="...",    # str optional
        )
        # z_dict keys: "satclip","geoclip","skysense","anygraph","tabfpn","llm"
    """

    def __init__(self, cfg: dict):
        super().__init__()
        device = cfg.get("device", "cpu")

        self.satclip  = SatCLIPEncoder(
            ckpt_path=cfg.get("satclip_ckpt"), device=device
        )
        self.geoclip  = GeoCLIPEncoder(device=device)
        self.skysense = SkySensePPEncoder(
            ckpt_path=cfg.get("skysense_ckpt"), device=device
        )
        self.anygraph = AnyGraphEncoder(
            ckpt_path=cfg.get("anygraph_ckpt"), device=device
        )
        self.tabpfn   = TabPFNEncoder(
            ckpt_path=cfg.get("tabpfn_ckpt"), device=device
        )
        self.llm      = LLMEncoder(
            model_name=cfg.get("llm_name", "Qwen/Qwen2.5-7B"), device=device
        )

        # Output dims per channel (used by bridges)
        self.out_dims = {
            "satclip":  SatCLIPEncoder.OUT_DIM,
            "geoclip":  GeoCLIPEncoder.OUT_DIM,
            "skysense": SkySensePPEncoder.OUT_DIM,
            "anygraph": AnyGraphEncoder.OUT_DIM,
            "tabfpn":   TabPFNEncoder.OUT_DIM,
            "llm":      LLMEncoder.OUT_DIM,
        }

    @torch.no_grad()
    def forward(
        self,
        coords: torch.Tensor,
        X_tab: torch.Tensor,
        y: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        subgraphs=None,
        prompt_text: str = "",
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict of raw frozen embeddings (no grad).

        coords    : (N, 2)
        X_tab     : (N, p)
        y         : (N,)     optional
        images    : (N,C,H,W) optional; if None, zeros are returned
        subgraphs : list[PyG Data] optional
        prompt_text: str shared across all N points
        """
        N = coords.shape[0]
        device = coords.device

        # --- Channel 1: SatCLIP ---
        z_satclip = self.satclip(coords)          # (N, 512)

        # --- Channel 2: GeoCLIP ---
        z_geoclip = self.geoclip(coords)          # (N, 512)

        # --- Channel 3: SkySense++ ---
        if images is not None:
            z_skysense = self.skysense(images)    # (N, 768)
        else:
            z_skysense = torch.zeros(N, SkySensePPEncoder.OUT_DIM, device=device)

        # --- Channel 4: AnyGraph ---
        if subgraphs is not None:
            z_anygraph = self.anygraph(subgraphs, N, device)  # (N, 256)
        else:
            z_anygraph = torch.zeros(N, AnyGraphEncoder.OUT_DIM, device=device)

        # --- Channel 5: TabPFN ---
        z_tabfpn = self.tabpfn(X_tab, y)         # (N, 512)

        # --- Channel 6: LLM ---
        z_llm = self.llm(prompt_text, N, device) # (N, 1024)

        return {
            "satclip":  z_satclip,
            "geoclip":  z_geoclip,
            "skysense": z_skysense,
            "anygraph": z_anygraph,
            "tabfpn":   z_tabfpn,
            "llm":      z_llm,
        }
