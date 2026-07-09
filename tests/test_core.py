"""Tests for regime_shap.core.RegimeSHAPAnalyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from sklearn.ensemble import GradientBoostingRegressor

from regime_shap import RegimeSHAPAnalyzer


@pytest.fixture
def analyzer():
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = X["a"] * 2 - X["b"] + rng.normal(scale=0.1, size=n)
    model = GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=0).fit(X, y)
    # three regimes; "shock" (5 rows) is below the default small-sample threshold of 10
    labels = ["calm"] * 20 + ["mid"] * 15 + ["shock"] * 5
    return RegimeSHAPAnalyzer(model, X, labels)


def test_import_from_top_level():
    from regime_shap import RegimeSHAPAnalyzer as Imported

    assert Imported is RegimeSHAPAnalyzer


def test_global_importance(analyzer):
    imp = analyzer.global_importance()
    assert isinstance(imp, pd.Series)
    assert set(imp.index) == {"a", "b", "c"}
    # "a" has the largest coefficient in the synthetic target, so it leads
    assert imp.idxmax() == "a"


def test_per_regime_shapes(analyzer):
    imp = analyzer.per_regime_importance()
    assert list(imp.columns) == ["calm", "mid", "shock"]
    assert list(imp.index) == ["a", "b", "c"]
    ranks = analyzer.per_regime_rankings()
    assert ranks.shape == imp.shape


def test_stability_matrix_is_square_and_symmetric(analyzer):
    m = analyzer.stability_matrix()
    assert m.shape == (3, 3)
    np.testing.assert_allclose(m.to_numpy(), m.to_numpy().T)


def test_stability_classified_has_pairs_and_bands(analyzer):
    classified = analyzer.stability_classified()
    assert len(classified) == 3  # 3 regimes -> 3 pairs
    assert set(classified["band"]).issubset({"stable", "moderately stable", "unstable"})
    assert classified["involves_small_regime"].any()  # "shock" is small


def test_small_regimes_and_sample_sizes(analyzer):
    assert analyzer.small_regimes() == {"shock"}
    sizes = analyzer.sample_sizes().set_index("regime")
    assert sizes.loc["shock", "n_observations"] == 5
    assert bool(sizes.loc["calm", "small_sample"]) is False


def test_bootstrap_cis_only_for_small_regime_pairs(analyzer):
    cis = analyzer.bootstrap_cis(n_bootstrap=20)
    # only pairs involving "shock": (calm, shock) and (mid, shock)
    assert len(cis) == 2
    pairs = set(zip(cis["regime_a"], cis["regime_b"]))
    assert ("calm", "shock") in pairs
    assert ("mid", "shock") in pairs
    assert (cis["ci_lower"] <= cis["ci_upper"]).all()
    assert (cis["effective_n"] == 5).all()


def test_plots_return_figures(analyzer):
    assert isinstance(analyzer.plot_global(), Figure)
    assert isinstance(analyzer.plot_per_regime(), Figure)
    assert isinstance(analyzer.plot_stability(), Figure)


def test_reports(analyzer, tmp_path):
    html = analyzer.to_html(title="Test")
    assert html.startswith("<!doctype html>")
    assert "stability_matrix" in html
    paths = analyzer.save_csv(tmp_path)
    assert (tmp_path / "stability_matrix.csv").exists()
    assert len(paths) == len(analyzer.results())


def test_empty_X_raises():
    model = GradientBoostingRegressor().fit(
        pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]}), [0.0, 1.0]
    )
    with pytest.raises(ValueError, match="X is empty"):
        RegimeSHAPAnalyzer(model, pd.DataFrame(columns=["a", "b"]), [])
