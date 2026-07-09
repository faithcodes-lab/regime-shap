"""Tests for regime_shap.breaks."""

from __future__ import annotations

import pandas as pd
import pytest

from regime_shap.breaks import regimes_to_labels


def _frame(n=6, dated=False):
    idx = pd.date_range("2020-01-01", periods=n, freq="QS") if dated else range(n)
    return pd.DataFrame({"a": range(n)}, index=idx)


def test_per_row_labels_passthrough():
    X = _frame(4)
    labels = regimes_to_labels(X, ["calm", "calm", "shock", "shock"])
    assert list(labels) == ["calm", "calm", "shock", "shock"]
    assert list(labels.index) == list(X.index)


def test_per_row_labels_wrong_length_raises():
    X = _frame(4)
    with pytest.raises(ValueError, match="3 labels but X has 4 rows"):
        regimes_to_labels(X, ["a", "b", "c"])


def test_per_row_labels_reject_missing():
    X = _frame(3)
    with pytest.raises(ValueError, match="must not contain missing"):
        regimes_to_labels(X, ["a", None, "b"])


def test_date_range_tuple_list_labels_by_position():
    X = _frame(4, dated=True)  # 2020Q1..2020Q4
    ranges = [("2020-01-01", "2020-03-31"), ("2020-04-01", "2020-12-31")]
    labels = regimes_to_labels(X, ranges)
    assert list(labels) == ["regime_1", "regime_2", "regime_2", "regime_2"]


def test_date_range_dict_labels():
    X = _frame(4, dated=True)
    ranges = {"pre": ("2020-01-01", "2020-06-30"), "post": ("2020-07-01", "2020-12-31")}
    labels = regimes_to_labels(X, ranges)
    assert list(labels) == ["pre", "pre", "post", "post"]


def test_date_range_uses_date_column_when_index_not_datetime():
    X = pd.DataFrame(
        {"a": range(3), "when": pd.to_datetime(["2021-01-01", "2021-06-01", "2021-12-01"])}
    )
    labels = regimes_to_labels(
        X,
        {"h1": ("2021-01-01", "2021-06-30"), "h2": ("2021-07-01", "2021-12-31")},
        date_column="when",
    )
    assert list(labels) == ["h1", "h1", "h2"]


def test_date_range_without_dates_raises():
    X = _frame(3, dated=False)  # integer index, no date column
    with pytest.raises(ValueError, match="datetime-indexed or a date_column"):
        regimes_to_labels(X, [("2020-01-01", "2020-12-31")])


def test_row_outside_all_ranges_raises():
    X = _frame(4, dated=True)  # through 2020Q4
    with pytest.raises(ValueError, match="fall outside every supplied regime range"):
        regimes_to_labels(X, [("2020-01-01", "2020-06-30")])  # only covers first 2 rows


def test_missing_date_column_raises():
    X = _frame(3, dated=True)
    with pytest.raises(ValueError, match="date_column 'nope' not found"):
        regimes_to_labels(X, {"r": ("2020-01-01", "2020-12-31")}, date_column="nope")
