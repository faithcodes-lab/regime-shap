"""Tests for regime_shap.compare."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_shap.compare import per_regime_importance, per_regime_rankings, regime_sample_sizes

# Two features, four rows: feature "a" carries all the weight, "b" none, in
# both regimes. Rows 0-1 are regime "calm", rows 2-3 are regime "shock".
_SHAP = np.array([[2.0, 0.0], [-4.0, 0.0], [1.0, 0.0], [-3.0, 0.0]])
_NAMES = ["a", "b"]
_LABELS = pd.Series(["calm", "calm", "shock", "shock"])


def test_per_regime_importance_mean_abs():
    imp = per_regime_importance(_SHAP, _NAMES, _LABELS)
    assert list(imp.columns) == ["calm", "shock"]
    assert list(imp.index) == ["a", "b"]
    assert imp.loc["a", "calm"] == pytest.approx(3.0)  # mean(|2|,|-4|)
    assert imp.loc["a", "shock"] == pytest.approx(2.0)  # mean(|1|,|-3|)
    assert imp.loc["b", "calm"] == pytest.approx(0.0)


def test_per_regime_rankings_top_feature_is_rank_one():
    imp = per_regime_importance(_SHAP, _NAMES, _LABELS)
    ranks = per_regime_rankings(imp)
    assert (ranks.loc["a"] == 1.0).all()
    assert (ranks.loc["b"] == 2.0).all()


def test_regime_sample_sizes_and_small_flag():
    sizes = regime_sample_sizes(_LABELS, small_sample_threshold=3)
    row = sizes.set_index("regime")
    assert row.loc["calm", "n_observations"] == 2
    assert bool(row.loc["calm", "small_sample"]) is True  # 2 < 3
    sizes_hi = regime_sample_sizes(_LABELS, small_sample_threshold=2)
    assert bool(sizes_hi.set_index("regime").loc["calm", "small_sample"]) is False  # 2 not < 2


def test_regime_order_is_first_appearance():
    labels = pd.Series(["shock", "calm", "shock", "calm"])
    shap = np.array([[1.0], [1.0], [1.0], [1.0]])
    imp = per_regime_importance(shap, ["a"], labels)
    assert list(imp.columns) == ["shock", "calm"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="labels has 3 entries but shap_values has 4 rows"):
        per_regime_importance(_SHAP, _NAMES, pd.Series(["a", "b", "c"]))


def test_feature_name_mismatch_raises():
    with pytest.raises(ValueError, match="feature_names has 1 names but shap_values has 2 columns"):
        per_regime_importance(_SHAP, ["a"], _LABELS)


def test_non_2d_raises():
    with pytest.raises(ValueError, match="must be 2D"):
        per_regime_importance(np.array([1.0, 2.0, 3.0, 4.0]), _NAMES, _LABELS)


def test_single_observation_regime_raises():
    labels = pd.Series(["calm", "calm", "calm", "lonely"])
    with pytest.raises(ValueError, match="regime 'lonely' has 1 observation"):
        per_regime_importance(_SHAP, _NAMES, labels)
