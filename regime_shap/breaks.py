"""Turn regime specifications into per-row regime labels.

Per-row labels are the package's internal representation: a label for every
row of the feature matrix, saying which regime that row belongs to. Users can
supply labels directly (the most general, domain-agnostic form) or, for
date-indexed data, give date ranges as a list of (start, end) tuples or a
{label: (start, end)} dict, which the adapters here convert to per-row labels.

Optionally, when the user has no regimes in mind, ``detect_breaks`` wraps the
``ruptures`` change-point library to propose break points from a series;
``ruptures`` is an optional dependency (install with the ``detection`` extra).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

DateRange = tuple[object, object]


def regimes_to_labels(
    X: pd.DataFrame, regimes: object, *, date_column: str | None = None
) -> pd.Series:
    """Resolve any accepted regime specification into a per-row label Series.

    Args:
        X: The feature matrix whose rows are being labelled.
        regimes: One of three forms:
            - a per-row sequence of labels, length ``len(X)`` (most general);
            - a list of ``(start, end)`` date ranges (labelled by position);
            - a ``{label: (start, end)}`` mapping.
            The date-range forms require ``X`` to be datetime-indexed or to have
            a ``date_column``.
        date_column: Name of a datetime column to use instead of the index for
            the date-range forms. Ignored for per-row labels.

    Returns:
        A Series of regime labels aligned to ``X`` (same length and index).

    Raises:
        ValueError: If a per-row sequence has the wrong length, if a date-range
            form is given without usable dates, or if a row falls outside every
            supplied range.
    """
    if isinstance(regimes, Mapping):
        return _ranges_to_labels(X, dict(regimes), date_column=date_column)
    if _is_date_range_list(regimes):
        labelled = {f"regime_{i + 1}": rng for i, rng in enumerate(regimes)}
        return _ranges_to_labels(X, labelled, date_column=date_column)
    return _per_row_labels(X, regimes)


def _per_row_labels(X: pd.DataFrame, labels: object) -> pd.Series:
    values = list(labels)
    if len(values) != len(X):
        raise ValueError(f"regimes has {len(values)} labels but X has {len(X)} rows")
    series = pd.Series(values, index=X.index)
    if series.isna().any():
        raise ValueError("per-row regime labels must not contain missing values")
    return series


def _is_date_range_list(regimes: object) -> bool:
    """A list/tuple whose items are all length-2 (start, end) pairs."""
    if isinstance(regimes, (str, bytes)) or not isinstance(regimes, Sequence):
        return False
    if len(regimes) == 0:
        return False
    return all(isinstance(item, (tuple, list)) and len(item) == 2 for item in regimes)


def _dates_from(X: pd.DataFrame, date_column: str | None) -> pd.DatetimeIndex:
    if date_column is not None:
        if date_column not in X.columns:
            raise ValueError(f"date_column '{date_column}' not found in X")
        return pd.DatetimeIndex(pd.to_datetime(X[date_column]))
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError(
            "date-range regimes require X to be datetime-indexed or a date_column to be given"
        )
    return X.index


def _ranges_to_labels(
    X: pd.DataFrame, ranges: Mapping[str, DateRange], *, date_column: str | None
) -> pd.Series:
    dates = _dates_from(X, date_column).to_numpy()
    labels = pd.Series([pd.NA] * len(X), index=X.index, dtype="object")
    for label, (start, end) in ranges.items():
        start_ts = np.datetime64(pd.Timestamp(start))
        end_ts = np.datetime64(pd.Timestamp(end))
        mask = (dates >= start_ts) & (dates <= end_ts)
        labels.iloc[np.where(mask)[0]] = label
    if labels.isna().any():
        n = int(labels.isna().sum())
        raise ValueError(f"{n} row(s) fall outside every supplied regime range")
    return labels


def detect_breaks(series: pd.Series, *, n_breaks: int = 3, model: str = "l2") -> list:
    """Propose change-point positions in a 1D series using ruptures (optional dependency).

    Args:
        series: The 1D series to search for structural breaks.
        n_breaks: Number of break points to return (uses ruptures' Dynp).
        model: The ruptures cost model ("l2", "l1", "rbf", ...).

    Returns:
        A list of break-point row positions (integer indices into ``series``),
        excluding the final endpoint.

    Raises:
        ImportError: If ruptures is not installed (pip install regime-shap[detection]).
    """
    try:
        import ruptures
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "detect_breaks requires the 'ruptures' package. "
            "Install it with: pip install regime-shap[detection]"
        ) from exc

    values = series.to_numpy().reshape(-1, 1)
    algo = ruptures.Dynp(model=model).fit(values)
    breakpoints = algo.predict(n_bkps=n_breaks)
    return [bp for bp in breakpoints if bp < len(series)]
