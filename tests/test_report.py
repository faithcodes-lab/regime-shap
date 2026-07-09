"""Tests for regime_shap.report."""

from __future__ import annotations

import pandas as pd

from regime_shap.report import save_csv, to_dict, to_html

_TABLES = {
    "global_importance": pd.Series({"a": 0.8, "b": 0.2}, name="mean_abs_shap"),
    "stability_matrix": pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], index=["calm", "shock"], columns=["calm", "shock"]
    ),
    "sample_sizes": pd.DataFrame(
        {"regime": ["calm", "shock"], "n_observations": [30, 6], "small_sample": [False, True]}
    ),
}


def test_to_dict_round_trips_values():
    d = to_dict(_TABLES)
    assert set(d) == {"global_importance", "stability_matrix", "sample_sizes"}
    assert d["global_importance"]["a"] == 0.8
    assert d["stability_matrix"]["calm"]["shock"] == 0.5


def test_save_csv_writes_each_table(tmp_path):
    paths = save_csv(_TABLES, tmp_path)
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
    reloaded = pd.read_csv(tmp_path / "stability_matrix.csv", index_col=0)
    assert reloaded.loc["calm", "shock"] == 0.5


def test_to_html_is_self_contained_and_includes_content():
    html = to_html(_TABLES, title="My report", notes=["small regimes flagged"])
    assert html.startswith("<!doctype html>")
    assert "<title>My report</title>" in html
    assert "My report" in html
    assert "small regimes flagged" in html
    # each table name appears as a heading, and a table is rendered
    for name in _TABLES:
        assert f"<h2>{name}</h2>" in html
    assert "<table" in html
    assert "0.8000" in html  # float formatting applied


def test_to_html_without_notes_omits_notes_list():
    html = to_html({"global_importance": _TABLES["global_importance"]})
    assert "<ul class='notes'>" not in html
