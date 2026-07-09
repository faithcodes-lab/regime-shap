"""Tests for regime_shap.stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from regime_shap.stability import (
    bootstrap_rankings,
    bootstrap_spearman_ci,
    classify_stability,
    pairwise_spearman_matrix,
)


def test_pairwise_identical_rankings_give_one():
    r = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4]}, index=["f1", "f2", "f3", "f4"])
    m = pairwise_spearman_matrix(r)
    assert m.loc["x", "y"] == pytest.approx(1.0)
    assert m.loc["x", "x"] == pytest.approx(1.0)


def test_pairwise_reversed_rankings_give_minus_one():
    r = pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1]}, index=["f1", "f2", "f3", "f4"])
    assert pairwise_spearman_matrix(r).loc["x", "y"] == pytest.approx(-1.0)


def test_pairwise_matrix_is_symmetric():
    r = pd.DataFrame({"a": [1, 2, 3], "b": [2, 1, 3], "c": [3, 2, 1]}, index=["f1", "f2", "f3"])
    m = pairwise_spearman_matrix(r)
    np.testing.assert_allclose(m.to_numpy(), m.to_numpy().T)


@pytest.mark.parametrize(
    "rho,expected",
    [
        (0.9, "stable"),
        (0.61, "stable"),
        (0.6, "moderately stable"),  # 0.6 is the top of the moderate band
        (0.45, "moderately stable"),
        (0.31, "moderately stable"),
        (0.3, "unstable"),  # 0.3 is the top of the unstable band
        (0.0, "unstable"),
        (-0.5, "unstable"),
    ],
)
def test_classify_stability_akoglu_bands(rho, expected):
    assert classify_stability(rho) == expected


def test_classify_stability_custom_thresholds():
    assert classify_stability(0.5, moderate_threshold=0.2, stable_threshold=0.4) == "stable"


@pytest.fixture
def small_regime_model():
    rng = np.random.default_rng(0)
    n = 6  # mirrors the GDP GFC/COVID regime size
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = X["a"] * 2 - X["b"] + rng.normal(scale=0.1, size=n)
    model = GradientBoostingRegressor(n_estimators=20, max_depth=2, random_state=0).fit(X, y)
    return model, X


def test_bootstrap_rankings_shape(small_regime_model):
    model, X = small_regime_model
    out = bootstrap_rankings(model, X, n_bootstrap=25, random_state=42)
    assert out.shape == (25, 3)


def test_bootstrap_rankings_deterministic(small_regime_model):
    model, X = small_regime_model
    a = bootstrap_rankings(model, X, n_bootstrap=25, random_state=42)
    b = bootstrap_rankings(model, X, n_bootstrap=25, random_state=42)
    np.testing.assert_array_equal(a, b)


def test_bootstrap_warns_for_tiny_regime():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(3, 2)), columns=["a", "b"])
    y = X["a"] + rng.normal(scale=0.1, size=3)
    model = GradientBoostingRegressor(n_estimators=10, max_depth=1, random_state=0).fit(X, y)
    with pytest.warns(UserWarning, match="bootstrap confidence intervals will be unreliable"):
        bootstrap_rankings(model, X, n_bootstrap=10)


def test_bootstrap_no_warning_at_n_six(small_regime_model):
    model, X = small_regime_model  # n = 6, the GDP small-regime size, should NOT warn
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bootstrap_rankings(model, X, n_bootstrap=10)


def test_bootstrap_spearman_ci_identical_is_one_one(small_regime_model):
    model, X = small_regime_model
    r = bootstrap_rankings(model, X, n_bootstrap=25, random_state=42)
    lower, upper = bootstrap_spearman_ci(r, r, n_bootstrap=25)
    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(1.0)


def test_bootstrap_spearman_ci_bounds(small_regime_model):
    model, X = small_regime_model
    a = bootstrap_rankings(model, X, n_bootstrap=25, random_state=42)
    b = bootstrap_rankings(model, X, n_bootstrap=25, random_state=7)
    lower, upper = bootstrap_spearman_ci(a, b, n_bootstrap=25)
    assert lower <= upper
    assert -1.0 <= lower <= 1.0
    assert -1.0 <= upper <= 1.0
