"""
gwf/utils/visualization.py
===========================
Spatial visualisation utilities for GWF coefficient analysis.

Requires: matplotlib, numpy, (optionally) scikit-learn for PCA/t-SNE.

Functions:
  plot_coefficient_map    — scatter plot coloured by sigma PCA component
  plot_effective_rank_map — scatter plot coloured by effective rank
  plot_modality_importance — 6-panel scatter plot of modality attention weights
  plot_uncertainty_map    — scatter plot coloured by prediction std (V2)
  plot_prediction_error   — scatter plot coloured by |y_pred - y_true|

Load analysis data with:
    data = np.load("gwf_analysis.npz")
"""

import numpy as np


def _check_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for visualisation. "
                          "Install with: pip install matplotlib")


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Reduce (N, d) → (N, 2) using PCA."""
    try:
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(X)
    except ImportError:
        # Manual PCA using numpy SVD
        X_c = X - X.mean(0)
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        return X_c @ Vt[:2].T


def plot_coefficient_map(
    coords:   np.ndarray,    # (N, 2) lat/lon
    sigma:    np.ndarray,    # (N, r) coefficient magnitudes
    title:    str = "Spatial Coefficient Map (PCA of sigma)",
    figsize:  tuple = (10, 7),
    save_path: str | None = None,
):
    """
    Visualise how regression behaviour varies spatially.
    Uses PCA to reduce sigma → 2D, colours each point by PC1.

    Regions with similar colour have similar regression behaviour.
    """
    plt = _check_matplotlib()
    pc = _pca_2d(sigma)           # (N, 2)
    c  = pc[:, 0]                 # colour by first principal component

    lons = coords[:, 1]
    lats = coords[:, 0]

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(lons, lats, c=c, cmap="RdYlBu", s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="PC1 of sigma")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_effective_rank_map(
    coords:    np.ndarray,   # (N, 2)
    eff_rank:  np.ndarray,   # (N,)
    title:     str = "Effective Rank Map",
    figsize:   tuple = (10, 7),
    save_path: str | None = None,
):
    """
    Visualise regression complexity spatially.
    High effective rank → many components active → complex local relationship.
    Low effective rank  → dominated by one component → simpler relationship.
    """
    plt = _check_matplotlib()
    lons, lats = coords[:, 1], coords[:, 0]

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(lons, lats, c=eff_rank, cmap="plasma", s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Effective rank")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_modality_importance(
    coords:    np.ndarray,   # (N, 2)
    attn_w:    np.ndarray,   # (N, 6) attention weights
    figsize:   tuple = (18, 9),
    save_path: str | None = None,
):
    """
    6-panel spatial map showing how important each modality is at each point.
    Higher weight → that encoder contributed more to the fused embedding.
    """
    plt = _check_matplotlib()
    channel_names = ["SatCLIP", "GeoCLIP", "SkySense++", "AnyGraph", "TabPFN", "LLM"]
    lons, lats = coords[:, 1], coords[:, 0]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i, (name, ax) in enumerate(zip(channel_names, axes)):
        sc = ax.scatter(lons, lats, c=attn_w[:, i],
                        cmap="Oranges", s=8, alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="Attention weight")
        ax.set_title(f"{name} importance")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.suptitle("Per-Modality Attention Weights (CrossChannelFusion)", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_prediction_error(
    coords:    np.ndarray,   # (N, 2)
    y_pred:    np.ndarray,   # (N,)
    y_true:    np.ndarray,   # (N,)
    figsize:   tuple = (10, 7),
    save_path: str | None = None,
):
    """
    Spatial map of absolute prediction error.
    Dark red spots indicate where the model struggles spatially.
    """
    plt = _check_matplotlib()
    error = np.abs(y_pred - y_true)
    lons, lats = coords[:, 1], coords[:, 0]

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(lons, lats, c=error, cmap="hot_r", s=10, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="|y_pred - y_true|")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Distribution of Prediction Error")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_uncertainty_map(
    coords:    np.ndarray,   # (N, 2)
    y_std:     np.ndarray,   # (N,)   — prediction std from MC passes
    figsize:   tuple = (10, 7),
    save_path: str | None = None,
):
    """
    V2 only: spatial uncertainty map. Bright regions = uncertain predictions.
    """
    plt = _check_matplotlib()
    lons, lats = coords[:, 1], coords[:, 0]

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(lons, lats, c=y_std, cmap="YlOrRd", s=10, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Prediction std (uncertainty)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Uncertainty Map (V2)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_all(
    analysis_path:    str = "gwf_analysis.npz",
    uncertainty_path: str | None = None,
    save_dir:         str | None = None,
):
    """
    Convenience function: load gwf_analysis.npz and plot all maps.

    Args:
        analysis_path    : path to gwf_analysis.npz (from evaluate.py)
        uncertainty_path : path to gwf_uncertainty.npz (V2 only, optional)
        save_dir         : if set, save all plots to this directory
    """
    import os
    data = np.load(analysis_path)
    coords   = data["coords"]
    sigma    = data["sigma"]
    eff_rank = data["eff_rank"]
    attn_w   = data["attn_w"]
    y_pred   = data["y_pred"]
    y_true   = data["y_true"]

    def _save(name):
        return os.path.join(save_dir, name) if save_dir else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plot_coefficient_map(coords, sigma,
                         save_path=_save("coefficient_map.png"))
    plot_effective_rank_map(coords, eff_rank,
                            save_path=_save("effective_rank_map.png"))
    plot_modality_importance(coords, attn_w,
                             save_path=_save("modality_importance.png"))
    plot_prediction_error(coords, y_pred, y_true,
                          save_path=_save("prediction_error.png"))

    if uncertainty_path:
        try:
            udata = np.load(uncertainty_path)
            plot_uncertainty_map(coords, udata["y_std"],
                                 save_path=_save("uncertainty_map.png"))
        except Exception as e:
            print(f"Could not plot uncertainty: {e}")
