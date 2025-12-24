# -*- coding: utf-8 -*-
"""
Figure utilities for SHAP-based interpretation.

Backward-compatibility policy:
- Function names, signatures, and defaults are preserved.
- Column names and expected SHAP array layout remain unchanged.

These helpers are designed for manuscript figure reproduction, not as a general
plotting library.
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_summary(shap_values, X_test, filename="shap_summary.png"):
    """
    Save a SHAP summary (beeswarm) plot.

    Parameters
    ----------
    shap_values
        SHAP values array aligned with X_test.
    X_test
        Feature matrix (pandas DataFrame).
    filename
        Output path.
    """
    plt.close("all")
    plt.figure(figsize=(12, 9), dpi=300)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.xlim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_dependence(
    shap_values,
    X_test,
    feature_name,
    filename="shap_dependence.png",
    interaction_index=None,     # kept for backward compatibility (unused by design)
    display_features=None,
    *,
    # --- speed knobs ---
    scatter_max_points=20000,
    scatter_rasterized=True,
    grid_points=200,

    # kNN smoother
    k_frac=0.1,
    k_min=200,
    agg="mean",
    trim_frac=0.02,

    # styling
    line_width=3,
    line_alpha=0.95,
    line_color="red",
    scatter_size=8,
    scatter_alpha=0.8,

    # axis limits
    xlim=None,   # e.g. (0, 1)
    ylim=None    # e.g. (-0.5, 0.5)
):
    """
    Plot a SHAP dependence plot with a local kNN smoothing curve.

    Notes
    -----
    - `interaction_index` is intentionally accepted but not used, to preserve
      the original function signature and external calls.
    - Subsampling uses a fixed seed for reproducibility.
    """
    plt.close("all")
    plt.figure(figsize=(9, 9), dpi=300)

    features_pass = display_features if display_features is not None else X_test

    # Resolve feature index
    if isinstance(feature_name, str):
        if feature_name not in features_pass.columns:
            raise KeyError(f"Feature '{feature_name}' was not found in the provided DataFrame.")
        feat_idx = features_pass.columns.get_loc(feature_name)
    else:
        feat_idx = int(feature_name)
        if feat_idx < 0 or feat_idx >= features_pass.shape[1]:
            raise IndexError(f"Feature index {feat_idx} is out of bounds for the input DataFrame.")

    x = np.asarray(features_pass.iloc[:, feat_idx], dtype=float)
    y = np.asarray(shap_values)[:, feat_idx].astype(float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size

    if n < 5:
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        return

    # Scatter with deterministic subsampling
    idx = np.arange(n)
    if n > scatter_max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, size=scatter_max_points, replace=False)

    plt.scatter(
        x[idx], y[idx],
        s=scatter_size,
        alpha=scatter_alpha,
        rasterized=scatter_rasterized
    )

    # Sort for smoothing
    order = np.argsort(x)
    xs, ys = x[order], y[order]

    # Local window size
    k = max(int(round(k_frac * n)), int(k_min))
    k = min(k, n)

    # Trim extremes
    tf = np.clip(trim_frac, 0.0, 0.49)
    qlo = np.quantile(xs, tf)
    qhi = np.quantile(xs, 1.0 - tf)

    g = np.linspace(qlo, qhi, int(grid_points))
    gi = np.searchsorted(xs, g, side="left")
    half = k // 2

    bx = np.empty_like(g)
    by = np.empty_like(g)

    for i, idx0 in enumerate(gi):
        lo = max(0, idx0 - half)
        hi = min(n, lo + k)
        lo = max(0, hi - k)
        xv, yv = xs[lo:hi], ys[lo:hi]
        if yv.size == 0:
            bx[i], by[i] = np.nan, np.nan
        else:
            if agg == "median":
                bx[i], by[i] = np.median(xv), np.median(yv)
            else:
                bx[i], by[i] = np.mean(xv), np.mean(yv)

    valid = np.isfinite(bx) & np.isfinite(by)
    if np.count_nonzero(valid) >= 2:
        plt.plot(
            bx[valid], by[valid],
            color=line_color,
            linewidth=line_width,
            alpha=line_alpha
        )

    # Reference line at y=0
    plt.axhline(0.0, color="black", linewidth=2, linestyle="--", alpha=0.7)

    # Axis limits
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # Styling (kept consistent with your original output)
    ax = plt.gca()
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, linestyle="--", linewidth=3, color="#e3e3e5", alpha=0.7)
    ax.set_facecolor("#f8f8ff")

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("SHAP value", fontsize=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_force(explainer, shap_values, X_test, index=0, filename="shap_force.png"):
    """
    Save a SHAP force plot for a single instance.
    """
    plt.close("all")
    plt.figure(figsize=(12, 9), dpi=300)
    shap.force_plot(
        explainer.expected_value,
        shap_values[index],
        X_test.iloc[index, :],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_waterfall(explainer, X_test, index=0, filename="shap_waterfall.png"):
    """
    Save a SHAP waterfall plot for a single instance.
    """
    plt.close("all")
    sv = explainer(X_test)
    shap.plots.waterfall(sv[index], show=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
