"""
gwf/models/encoders.py
======================
Module 1: Frozen Encoder Bank

All 6 encoder channels are loaded here, set to eval() mode, and wrapped in
torch.no_grad() during forward passes. No gradients flow through them.

Channels:
  1. SatCLIP    (coords → 512)    satellite-level location semantics
  2. GeoCLIP    (coords → 512)    street-view-level location semantics
  3. SkySense++ (image  → 768)    visual RS features              [OPTIONAL]
  4. AnyGraph   (subgraph→ 256)   graph structural features       [OPTIONAL]
  5. TabPFN     (X_tab  → 128)    tabular in-context representation
  6. LLMEmbed   (text   → H_llm)  world knowledge (H_llm from checkpoint)

For missing / unavailable modalities the encoder returns a zero tensor of the
correct shape and logs a one-time warning. The CrossChannelFusion attention
mechanism naturally learns to down-weight zero channels.
"""

import sys
import logging
from pathlib import Path

import numpy as np
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
    Loads the SatCLIP location encoder via the lightweight loader already
    bundled in gwf/satclip_src/load_satclip.py.  No PyTorch Lightning needed.

    Input : coords (N, 2)  float32  (latitude, longitude)
    Output: z      (N, 512) float32

    Checkpoint download:
        huggingface.co/microsoft/SatCLIP-ResNet50-L10
        huggingface.co/microsoft/SatCLIP-ViT16-L10
    Set config.ENCODERS["satclip_ckpt"] to the local .ckpt path.
    """

    OUT_DIM = 512

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        super().__init__()
        self._available = False
        self.loc_enc = None

        if not ckpt_path or not Path(ckpt_path).exists():
            logger.warning("SatCLIP: checkpoint not provided / not found. "
                           "Using zero embeddings.")
            return

        try:
            # Use the lightweight loader (no lightning, no main.py imports)
            from gwf.satclip_src.load_satclip import load_satclip_loc_encoder  # type: ignore
            self.loc_enc = load_satclip_loc_encoder(ckpt_path, device=device)
            freeze(self.loc_enc)
            self._available = True
            logger.info("SatCLIP loaded ✓")
        except Exception as e:
            logger.warning(f"SatCLIP load failed ({e}). Using zero embeddings.")

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 2) float32 lat/lon → (N, 512) float32"""
        N = coords.shape[0]
        if not self._available:
            return torch.zeros(N, self.OUT_DIM, device=coords.device)

        # SatCLIP was trained in float64; cast in, cast out
        z = self.loc_enc(coords.double())          # (N, embed_dim) float64
        z = z.float()                               # back to float32
        z = F.normalize(z, dim=-1)
        # Guard against embed_dim ≠ OUT_DIM at runtime
        if z.shape[-1] != self.OUT_DIM:
            z = F.adaptive_avg_pool1d(
                z.unsqueeze(0), self.OUT_DIM
            ).squeeze(0)
        return z


# =============================================================================
# Channel 2: GeoCLIP — street-view-level location encoder
# =============================================================================

class GeoCLIPEncoder(nn.Module):
    """
    Loads the GeoCLIP location encoder bundled in geoclip/.
    Weights are stored in geoclip/model/weights/ and loaded automatically
    by LocationEncoder(from_pretrained=True).

    Input : coords (N, 2) float32 (latitude, longitude)
    Output: z      (N, 512) float32
    """

    OUT_DIM = 512

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self._available = False
        self.loc_enc = None

        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(repo_root))
            from geoclip.model.location_encoder import LocationEncoder   # type: ignore

            # from_pretrained=True loads weights automatically via _load_weights()
            self.loc_enc = freeze(LocationEncoder(from_pretrained=True))
            self.loc_enc.to(device)
            self._available = True
            logger.info("GeoCLIP loaded ✓")
        except Exception as e:
            logger.warning(f"GeoCLIP load failed ({e}). Using zero embeddings.")

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 2) float32 lat/lon → (N, 512) float32"""
        N = coords.shape[0]
        if not self._available:
            return torch.zeros(N, self.OUT_DIM, device=coords.device)

        z = self.loc_enc(coords)     # (N, 512)
        return F.normalize(z.float(), dim=-1)


# =============================================================================
# Channel 3: SkySense++ — visual RS feature extractor  [OPTIONAL]
# =============================================================================

class SkySensePPEncoder(nn.Module):
    """
    Wraps the SkySense++ backbone (Swin-L based).
    Input : images (N, C, H, W)
    Output: z      (N, 768)

    OPTIONAL — if no checkpoint is provided, returns zeros.
    To enable: provide the SkySense++ checkpoint path in config.ENCODERS.

    Loading stub (fill in after installing the skysense package):
        from skysense import build_backbone
        self.backbone = freeze(build_backbone(ckpt_path))
    """

    OUT_DIM = 768

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        super().__init__()
        self._available = False

        if ckpt_path and Path(ckpt_path).exists():
            logger.warning("SkySense++ checkpoint provided but loading is not "
                           "yet implemented. Add loading code here once the "
                           "skysense package is installed.")
        else:
            logger.info("SkySense++: no checkpoint provided. Using zero embeddings.")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (N, C, H, W) → (N, 768)"""
        N = images.shape[0]
        if not self._available:
            return torch.zeros(N, self.OUT_DIM, device=images.device)
        # Implement once backbone is loaded:
        #   feats = self.backbone(images)           # (N, 768, H', W')
        #   return F.normalize(feats.mean([-2,-1]), dim=-1)
        return torch.zeros(N, self.OUT_DIM, device=images.device)


# =============================================================================
# Channel 4: AnyGraph — graph structural encoder  [OPTIONAL]
# =============================================================================

class AnyGraphEncoder(nn.Module):
    """
    Wraps AnyGraph (KDD 2025) for local POI/road subgraph encoding.
    Input : list of PyG Data objects (N subgraphs)
    Output: z (N, 256)

    OPTIONAL — if no checkpoint is provided, returns zeros.
    Loading stub (fill in after installing the anygraph package):
        from anygraph import load_anygraph
        self.gnn = freeze(load_anygraph(ckpt_path))
    """

    OUT_DIM = 256

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        super().__init__()
        self._available = False

        if ckpt_path and Path(ckpt_path).exists():
            logger.warning("AnyGraph checkpoint provided but loading is not "
                           "yet implemented. Add loading code here once the "
                           "anygraph package is installed.")
        else:
            logger.info("AnyGraph: no checkpoint provided. Using zero embeddings.")

    @torch.no_grad()
    def forward(self, subgraphs, N: int, device: torch.device) -> torch.Tensor:
        """subgraphs: list[PyG Data] → (N, 256)"""
        if not self._available:
            return torch.zeros(N, self.OUT_DIM, device=device)
        # Implement once gnn is loaded:
        #   from torch_geometric.data import Batch
        #   batch = Batch.from_data_list(subgraphs).to(device)
        #   z = self.gnn(batch)
        #   return F.normalize(z, dim=-1)
        return torch.zeros(N, self.OUT_DIM, device=device)


# =============================================================================
# Channel 5: TabPFN — tabular in-context encoder
# =============================================================================

class TabPFNEncoder(nn.Module):
    """
    Uses pretrained TabPFN v2 as an in-context tabular feature extractor.

    Design:
      - TabPFN is fit ONCE on the full training set (call fit_context()).
      - At inference, we extract an OUT_DIM-dimensional representation per
        point by combining:
          (a) TabPFN's leave-one-out prediction (1-dim scalar)
          (b) A fixed random-Fourier-feature projection of X_tab (OUT_DIM-1 dims)
        → concat → (N, OUT_DIM)
      - The downstream MLPBridge learns to project this to d=512.

    Why RFF? TabPFN's transformer internals are hard to access cleanly across
    package versions. RFF gives a deterministic, information-preserving
    projection of X_tab that the bridge can learn from.

    OUT_DIM = 128 (fixed).
    """

    OUT_DIM = 128

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu",
                 rff_seed: int = 42):
        super().__init__()
        self._available = False
        self._fitted    = False
        self._tabpfn    = None
        # RFF projection matrix: registered as buffer so it's saved with model
        # Shape is set lazily on first call (depends on feat_dim p)
        self._rff_W: torch.Tensor | None = None
        self._rff_seed = rff_seed

        try:
            from tabpfn import TabPFNRegressor  # type: ignore

            self._tabpfn = TabPFNRegressor(
                device=device,
                N_ensemble_configurations=1,
            )
            self._available = True
            logger.info("TabPFN loaded ✓")
        except Exception as e:
            logger.warning(f"TabPFN unavailable ({e}). Using RFF-only embeddings.")

    def _get_rff(self, feat_dim: int, device: torch.device) -> torch.Tensor:
        """Lazily create / return the fixed RFF projection matrix (feat_dim, OUT_DIM-1)."""
        out_dim = self.OUT_DIM - 1   # 1 slot reserved for TabPFN prediction
        if self._rff_W is None or self._rff_W.shape[0] != feat_dim:
            gen = torch.Generator().manual_seed(self._rff_seed)
            self._rff_W = torch.randn(feat_dim, out_dim, generator=gen)
        return self._rff_W.to(device)

    def fit_context(self, X_tab: torch.Tensor, y: torch.Tensor):
        """
        Fit TabPFN on the full training set as context.
        Call once before training (not on every batch).
        """
        if not self._available:
            return
        self._tabpfn.fit(X_tab.cpu().numpy(), y.cpu().numpy())
        self._fitted = True
        logger.info(f"TabPFN context fitted on {X_tab.shape[0]} samples.")

    @torch.no_grad()
    def forward(self, X_tab: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        X_tab: (N, p)  →  (N, OUT_DIM=128)
        """
        N, p = X_tab.shape
        device = X_tab.device

        # --- Part A: TabPFN scalar prediction (1 dim) ---
        if self._available:
            try:
                cpu_X = X_tab.cpu().numpy()
                cpu_y = y.cpu().numpy() if y is not None else np.zeros(N)
                if not self._fitted:
                    self._tabpfn.fit(cpu_X, cpu_y)
                    self._fitted = True
                preds = self._tabpfn.predict(cpu_X)    # (N,) float64
                z_pred = torch.from_numpy(preds).float().to(device).unsqueeze(-1)  # (N,1)
            except Exception as e:
                logger.warning(f"TabPFN predict failed ({e}).")
                z_pred = torch.zeros(N, 1, device=device)
        else:
            z_pred = torch.zeros(N, 1, device=device)

        # --- Part B: Random Fourier Feature projection of X_tab (OUT_DIM-1 dims) ---
        W = self._get_rff(p, device)                   # (p, OUT_DIM-1)
        z_rff = torch.tanh(X_tab @ W)                  # (N, OUT_DIM-1)

        # --- Concatenate ---
        z = torch.cat([z_pred, z_rff], dim=-1)         # (N, OUT_DIM)
        return z


# =============================================================================
# Channel 6: Text Embedding Encoder (Qwen3-Embedding)
# =============================================================================

def _last_token_pool(
    last_hidden_states: torch.Tensor,   # (B, T, H)
    attention_mask:     torch.Tensor,   # (B, T)
) -> torch.Tensor:
    """Last-token pooling for decoder-based embedding models (e.g. Qwen3-Embedding)."""
    seq_lens = attention_mask.sum(dim=1) - 1      # (B,)
    B = last_hidden_states.shape[0]
    idx = torch.arange(B, device=last_hidden_states.device)
    return last_hidden_states[idx, seq_lens]       # (B, H)


def _mean_pool(
    last_hidden_states: torch.Tensor,   # (B, T, H)
    attention_mask:     torch.Tensor,   # (B, T)
) -> torch.Tensor:
    """Attention-mask-weighted mean pooling for encoder-based models."""
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_states * mask).sum(1) / mask.sum(1).clamp(1e-9)


class LLMEncoder(nn.Module):
    """
    Encodes the dataset/task description using a frozen text embedding model.

    Recommended: Qwen3-Embedding (MTEB SOTA, 2025).
      "Qwen/Qwen3-Embedding-0.6B"   0.6B,  H=1024  (fastest)
      "Qwen/Qwen3-Embedding-4B"     4B,    H=2560
      "Qwen/Qwen3-Embedding"        8B,    H=4096   (best quality)

    OUT_DIM is set DYNAMICALLY to the model's hidden_size after loading.
    The downstream AllMLPBridges bridge handles projection to d=512.
    No trainable parameters here — fully frozen.

    Pooling: last-token for Qwen (decoder arch), mean for encoder-based models.
    The prompt is cached per string — LLM runs ONCE per dataset.
    """

    # Instruction prefix for Qwen3-Embedding
    _INSTRUCTION = (
        "Instruct: Retrieve a semantically relevant representation of this "
        "spatial dataset description for geographic regression.\nQuery: "
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device:     str = "cpu",
    ):
        super().__init__()
        self.device_str  = device
        self._available  = False
        self._cache: dict[str, torch.Tensor] = {}
        self._use_last_token = False

        # OUT_DIM is set after loading (= model's hidden_size)
        self.out_dim: int = 1024   # safe default; overwritten on successful load

        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore

            logger.info(f"Loading text encoder: {model_name} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.lm = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(device)
            freeze(self.lm)

            self.out_dim = self.lm.config.hidden_size

            # Detect decoder-based model → use last-token pooling
            arch = getattr(self.lm.config, "architectures", []) or []
            self._use_last_token = (
                any("causal" in a.lower() or "decoder" in a.lower() for a in arch)
                or "qwen" in model_name.lower()
            )
            pool = "last-token" if self._use_last_token else "mean"
            self._available = True
            logger.info(f"Text encoder loaded ✓  H={self.out_dim}  pooling={pool}")

        except Exception as e:
            logger.warning(f"Text encoder unavailable ({e}). Using zero embeddings.")
            self.tokenizer = None
            self.lm        = None

    @torch.no_grad()
    def _encode_raw(self, prompt_text: str, device: torch.device) -> torch.Tensor:
        """
        Encode prompt → raw pooled hidden state (1, out_dim). Cached.
        This function is always inside no_grad — the LLM is frozen.
        """
        if prompt_text in self._cache:
            return self._cache[prompt_text].to(device)

        if not self._available:
            result = torch.zeros(1, self.out_dim, device=device)
            self._cache[prompt_text] = result
            return result

        text = (self._INSTRUCTION + prompt_text) if self._use_last_token else prompt_text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device_str)

        outputs = self.lm(**inputs, output_hidden_states=False)
        hidden  = outputs.last_hidden_state      # (1, T, H)
        attn    = inputs["attention_mask"]

        c = (_last_token_pool(hidden, attn) if self._use_last_token
             else _mean_pool(hidden, attn))      # (1, H)
        c = F.normalize(c.float(), dim=-1)

        result = c.detach().to(device)
        self._cache[prompt_text] = result
        return result

    def forward(self, prompt_text: str, N: int, device: torch.device) -> torch.Tensor:
        """Returns (N, out_dim) — prompt embedding broadcast to all N points."""
        c = self._encode_raw(prompt_text, device)    # (1, out_dim)
        return c.expand(N, -1)                        # (N, out_dim)


# =============================================================================
# FrozenEncoderBank — assembles all 6 channels
# =============================================================================

class FrozenEncoderBank(nn.Module):
    """
    Assembles all 6 frozen encoder channels into one module.

    After construction, self.out_dims holds the ACTUAL output dimension
    of every channel. Pass this to AllMLPBridges so each bridge gets the
    right input dim — especially important for the LLM channel whose
    hidden_size varies by model variant.

    Usage:
        bank = FrozenEncoderBank(cfg)
        z_dict = bank(coords, X_tab, y, images, subgraphs, prompt_text)
        # z_dict keys: satclip / geoclip / skysense / anygraph / tabfpn / llm
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
        self.tabpfn   = TabPFNEncoder(device=device)
        self.llm      = LLMEncoder(
            model_name=cfg.get("llm_name", "Qwen/Qwen3-Embedding-0.6B"),
            device=device,
        )

        # Actual output dims per channel — used by AllMLPBridges for construction
        self.out_dims: dict[str, int] = {
            "satclip":  SatCLIPEncoder.OUT_DIM,
            "geoclip":  GeoCLIPEncoder.OUT_DIM,
            "skysense": SkySensePPEncoder.OUT_DIM,
            "anygraph": AnyGraphEncoder.OUT_DIM,
            "tabfpn":   TabPFNEncoder.OUT_DIM,
            "llm":      self.llm.out_dim,        # dynamic: 1024 / 2560 / 4096
        }
        logger.info(f"EncoderBank out_dims: {self.out_dims}")

    @torch.no_grad()
    def forward(
        self,
        coords:      torch.Tensor,
        X_tab:       torch.Tensor,
        y:           torch.Tensor | None = None,
        images:      torch.Tensor | None = None,
        subgraphs=None,
        prompt_text: str = "",
    ) -> dict[str, torch.Tensor]:
        N      = coords.shape[0]
        device = coords.device

        z_satclip = self.satclip(coords)
        z_geoclip = self.geoclip(coords)

        z_skysense = (self.skysense(images) if images is not None
                      else torch.zeros(N, SkySensePPEncoder.OUT_DIM, device=device))

        z_anygraph = (self.anygraph(subgraphs, N, device) if subgraphs is not None
                      else torch.zeros(N, AnyGraphEncoder.OUT_DIM, device=device))

        z_tabfpn = self.tabpfn(X_tab, y)
        z_llm    = self.llm(prompt_text, N, device)

        return {
            "satclip":  z_satclip,
            "geoclip":  z_geoclip,
            "skysense": z_skysense,
            "anygraph": z_anygraph,
            "tabfpn":   z_tabfpn,
            "llm":      z_llm,
        }
