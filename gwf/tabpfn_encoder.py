"""
GWF — TabPFN In-Context Encoder

Wraps the pretrained TabPFN PerFeatureTransformer (frozen) to extract
pre-decoder hidden states as context-aware tabular embeddings.

For each query point i with k neighbours:
  Context (train) : (X_nbr, y_nbr)  — k neighbour (feature, label) pairs
  Test            : X_query          — 1 query feature vector

TabPFN processes this as an in-context learning problem and produces hidden
states at the last transformer layer BEFORE the MLP decoder head.

We extract (via only_return_standard_out=False):
  test_embeddings  → z_query : (B, ninp)   — query point pre-MLP embedding
  train_embeddings → z_nbr   : (B, k, ninp) — neighbour pre-MLP embeddings

These replace the lightweight InContextEncoder + cross-attention block.
TabPFN's weights are fully frozen; only downstream GWF modules are trained.

Loading
-------
Provide model_path to load from a local .ckpt file (recommended; avoids
network access):

    enc = TabPFNInContextEncoder(model_path="/path/to/tabpfn-v2-regressor.ckpt")

If model_path=None the constructor attempts a HuggingFace download.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tabpfn_transformer(model_path: str | None = None):
    """
    Return the frozen PerFeatureTransformer from a TabPFNRegressor.

    Parameters
    ----------
    model_path : str | None
        Path to a local TabPFN v2 regressor .ckpt file.
        If None, the model is downloaded from HuggingFace (requires internet).

    Returns
    -------
    model : PerFeatureTransformer
        The inner transformer (ALL parameters set to requires_grad=False).
    ninp : int
        Hidden dimension of the transformer (= model.ninp).
    """
    try:
        from tabpfn import TabPFNRegressor
        from tabpfn.model_loading import load_model_criterion_config
    except ImportError as e:
        raise ImportError(
            "tabpfn is required for TabPFNInContextEncoder. "
            "Install it with: pip install tabpfn"
        ) from e

    # load_model_criterion_config returns (model, criterion, config)
    model, _, _ = load_model_criterion_config(
        model_path=model_path,
        task_type="regression",
        inference_config=None,
        fit_mode="low_memory",
    )

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    return model, model.ninp


# ─────────────────────────────────────────────────────────────────────────────
# TabPFN In-Context Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TabPFNInContextEncoder(nn.Module):
    """
    In-context encoder backed by a frozen pretrained TabPFN transformer.

    Replaces InContextEncoder: produces context-aware embeddings for BOTH
    the query point and its k neighbours in a single TabPFN forward pass.

    TabPFN's PerFeatureTransformer sees:
      - k context rows : (X_nbr, y_nbr)
      - 1  test   row  : (X_query, NaN)

    With only_return_standard_out=False it returns hidden states from the
    last transformer block BEFORE the MLP decoder head.  These are the
    "pre-MLP" representations the user requested.

    Parameters
    ----------
    model_path : str | None
        Path to a local TabPFN v2 regressor .ckpt checkpoint.
        Pass None to attempt an automatic HuggingFace download.

    Input
    -----
    x_query : (B, p)
    x_nbr   : (B, k, p)
    y_nbr   : (B, k)      — neighbour labels (same scale as training targets)

    Output
    ------
    z_query : (B, emb_dim)    — query pre-MLP embedding
    z_nbr   : (B, k, emb_dim) — neighbour pre-MLP embeddings
    emb_dim == tabpfn.ninp
    """

    def __init__(self, model_path: str | None = None):
        super().__init__()
        model, ninp = _load_tabpfn_transformer(model_path)
        self.tabpfn = model
        self.emb_dim: int = ninp

    # ------------------------------------------------------------------

    def forward(
        self,
        x_query: torch.Tensor,   # (B, p)
        x_nbr:   torch.Tensor,   # (B, k, p)
        y_nbr:   torch.Tensor,   # (B, k)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pre-decoder hidden states from the frozen TabPFN transformer.

        The forward pass through TabPFN is run under torch.no_grad() because
        all TabPFN parameters are frozen.  Downstream learnable modules
        (NodeProjection, DynamicKernelGenerator) still receive gradients
        because their own parameters require grad.

        Returns
        -------
        z_query : (B, emb_dim)
        z_nbr   : (B, k, emb_dim)
        """
        B, k, p = x_nbr.shape

        # ── Build sequence input for TabPFN ──────────────────────────────────
        # PerFeatureTransformer expects x : (seq_len, batch_size, num_features)
        # We arrange: first k rows = context (neighbours), last row = test (query)
        x_ctx  = x_nbr.permute(1, 0, 2)            # (k,   B, p)
        x_test = x_query.unsqueeze(0)               # (1,   B, p)
        x_seq  = torch.cat([x_ctx, x_test], dim=0) # (k+1, B, p)

        # PerFeatureTransformer expects y : (seq_len_context, batch_size)
        # Only context labels are supplied; the model masks test labels to NaN.
        y_ctx = y_nbr.t()                           # (k, B)

        # ── TabPFN forward (frozen, no grad) ─────────────────────────────────
        with torch.no_grad():
            out = self.tabpfn(
                x_seq,
                y_ctx,
                only_return_standard_out=False,
            )

        # out["test_embeddings"]  : (B, n_test, ninp)  — n_test == 1
        # out["train_embeddings"] : (B, k,      ninp)
        z_query = out["test_embeddings"].squeeze(1)   # (B, ninp)
        z_nbr   = out["train_embeddings"]             # (B, k, ninp)

        return z_query, z_nbr
