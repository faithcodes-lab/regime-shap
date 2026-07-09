"""Figures for regime-aware SHAP analysis.

Each function returns a matplotlib Figure; the caller saves or displays it. The
figures are model- and domain-agnostic, they take the importance and stability
tables the other modules produce, not any project-specific data. The stability
heatmap colours cells by the Akoglu (2018) bands and can flag small-sample
regimes so a reader does not over-read their rows.
"""

from __future__ import annotations

from collections.abc import Set

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

from regime_shap.stability import DEFAULT_MODERATE_THRESHOLD, DEFAULT_STABLE_THRESHOLD

_UNSTABLE_COLOUR = "#d73027"
_MODERATE_COLOUR = "#fdae61"
_STABLE_COLOUR = "#1a9850"


def plot_global_importance(
    importance: pd.Series, *, title: str = "Global SHAP feature importance"
) -> Figure:
    """Horizontal bar chart of global mean absolute SHAP per feature.

    Args:
        importance: A Series of mean absolute SHAP indexed by feature.
        title: The figure title.

    Returns:
        A matplotlib Figure, most-important feature at the top.
    """
    ordered = importance.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(ordered.index.astype(str), ordered.to_numpy(), color="tab:blue")
    ax.set_xlabel("Mean absolute SHAP value", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_per_regime_importance(
    importance: pd.DataFrame, *, title: str = "Per-regime SHAP feature importance"
) -> Figure:
    """Heatmap of mean absolute SHAP per feature (rows) within each regime (columns).

    Args:
        importance: A feature-by-regime DataFrame of mean absolute SHAP.
        title: The figure title.

    Returns:
        A matplotlib Figure with a colour bar.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(importance.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(importance.columns)))
    ax.set_xticklabels([str(c) for c in importance.columns], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(importance.index)))
    ax.set_yticklabels([str(i) for i in importance.index], fontsize=8)
    ax.set_title(title, fontsize=13)
    fig.colorbar(image, ax=ax, label="Mean absolute SHAP value")
    fig.tight_layout()
    return fig


def plot_stability_heatmap(
    matrix: pd.DataFrame,
    *,
    small_regimes: Set[str] | None = None,
    moderate_threshold: float = DEFAULT_MODERATE_THRESHOLD,
    stable_threshold: float = DEFAULT_STABLE_THRESHOLD,
    title: str = "SHAP ranking stability across regimes (Spearman rho)",
) -> Figure:
    """Regime-by-regime Spearman heatmap coloured by the Akoglu (2018) bands.

    Args:
        matrix: A square regime-by-regime DataFrame of Spearman rho.
        small_regimes: Regime labels to flag with an asterisk (small samples,
            whose rows should be read with the bootstrap confidence intervals).
        moderate_threshold: Upper edge of the unstable band (default 0.3).
        stable_threshold: Upper edge of the moderate band (default 0.6).
        title: The figure title.

    Returns:
        A matplotlib Figure: cells coloured red (unstable), orange (moderate),
        or green (stable), annotated with rho to two decimals.
    """
    flagged = small_regimes or set()
    labels = [f"{r} *" if r in flagged else str(r) for r in matrix.columns]
    cmap = ListedColormap([_UNSTABLE_COLOUR, _MODERATE_COLOUR, _STABLE_COLOUR])
    norm = BoundaryNorm([-1.0, moderate_threshold, stable_threshold, 1.0], cmap.N)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix.to_numpy(), cmap=cmap, norm=norm)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix.to_numpy()[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
    ax.set_title(title, fontsize=13)
    cbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[
            (-1 + moderate_threshold) / 2,
            (moderate_threshold + stable_threshold) / 2,
            (stable_threshold + 1) / 2,
        ],
    )
    cbar.ax.set_yticklabels(
        [
            f"unstable\n(<= {moderate_threshold})",
            f"moderate\n({moderate_threshold} to {stable_threshold})",
            f"stable\n(> {stable_threshold})",
        ]
    )
    fig.tight_layout()
    return fig
