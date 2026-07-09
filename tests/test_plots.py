"""Tests for regime_shap.plots."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from regime_shap.plots import (
    plot_global_importance,
    plot_per_regime_importance,
    plot_stability_heatmap,
)


def test_global_importance_returns_figure_and_saves(tmp_path):
    imp = pd.Series({"a": 0.8, "b": 0.2, "c": 0.0}, name="mean_abs_shap")
    fig = plot_global_importance(imp)
    assert isinstance(fig, Figure)
    out = tmp_path / "global.png"
    fig.savefig(out, dpi=150)
    assert out.stat().st_size > 3000


def test_per_regime_importance_returns_figure(tmp_path):
    imp = pd.DataFrame({"calm": [0.8, 0.2], "shock": [1.4, 0.3]}, index=["a", "b"])
    fig = plot_per_regime_importance(imp)
    assert isinstance(fig, Figure)
    out = tmp_path / "per_regime.png"
    fig.savefig(out, dpi=150)
    assert out.stat().st_size > 3000


def test_stability_heatmap_flags_small_regimes():
    regimes = ["calm", "gfc", "covid"]
    matrix = pd.DataFrame(
        [[1.0, 0.9, 0.2], [0.9, 1.0, 0.1], [0.2, 0.1, 1.0]], index=regimes, columns=regimes
    )
    fig = plot_stability_heatmap(matrix, small_regimes={"gfc", "covid"})
    assert isinstance(fig, Figure)
    xlabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "gfc *" in xlabels
    assert "covid *" in xlabels
    assert "calm" in xlabels  # not flagged


def test_stability_heatmap_without_small_regimes():
    matrix = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["a", "b"], columns=["a", "b"])
    fig = plot_stability_heatmap(matrix)
    assert isinstance(fig, Figure)
    xlabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert xlabels == ["a", "b"]  # no asterisks
