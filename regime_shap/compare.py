"""Per-regime SHAP feature importance and rankings.

Given SHAP values and a per-row regime label, this computes, within each
regime, the mean absolute SHAP value per feature (that regime's feature
importance) and the resulting importance ranking. It also reports each
regime's sample size and flags small samples, whose rankings carry more
uncertainty. The functions operate on plain arrays, not a fitted model, so
they are model-agnostic and cheap to test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_SMALL_SAMPLE_THRESHOLD = 10


def per_regime_importance(
    shap_values: np.ndarray, feature_names: Sequence[str], labels: pd.Series
) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, within each regime.

    Args:
        shap_values: A 2D array of SHAP values, rows by features, aligned to
            the rows the labels describe.
        feature_names: Column names for the features, length must match the
            number of SHAP columns.
        labels: A per-row regime label for each row of ``shap_values``.

    Returns:
        A DataFrame indexed by feature, one column per regime (in first-
        appearance order), holding the regime's mean absolute SHAP per feature.

    Raises:
        ValueError: If shap_values is not 2D, if the lengths of labels or
            feature_names do not match, or if any regime has fewer than two
            observations (a single row cannot give a meaningful importance).
    """
    values = np.asarray(shap_values)
    labels = pd.Series(list(labels))
    if values.ndim != 2:
        raise ValueError(f"shap_values must be 2D (rows x features), got shape {values.shape}")
    if len(labels) != values.shape[0]:
        raise ValueError(
            f"labels has {len(labels)} entries but shap_values has {values.shape[0]} rows"
        )
    if values.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names has {len(feature_names)} names but shap_values has "
            f"{values.shape[1]} columns"
        )

    columns: dict[str, np.ndarray] = {}
    for regime in pd.unique(labels):
        mask = (labels == regime).to_numpy()
        n = int(mask.sum())
        if n < 2:
            raise ValueError(f"regime '{regime}' has {n} observation(s); at least 2 are required")
        columns[regime] = np.abs(values[mask]).mean(axis=0)
    return pd.DataFrame(columns, index=list(feature_names))


def per_regime_rankings(importance: pd.DataFrame) -> pd.DataFrame:
    """Rank features within each regime, 1 being the most important.

    Args:
        importance: A feature-by-regime DataFrame of mean absolute SHAP, as
            returned by :func:`per_regime_importance`.

    Returns:
        A DataFrame of the same shape holding ranks per regime; tied features
        share the average of the ranks they span.
    """
    return importance.rank(ascending=False, method="average")


def regime_sample_sizes(
    labels: pd.Series, *, small_sample_threshold: int = DEFAULT_SMALL_SAMPLE_THRESHOLD
) -> pd.DataFrame:
    """Per-regime observation counts with a small-sample flag.

    Args:
        labels: A per-row regime label.
        small_sample_threshold: Regimes with fewer observations than this are
            flagged; their rankings should be read with the bootstrap
            confidence intervals rather than at face value.

    Returns:
        A DataFrame with one row per regime: ``regime``, ``n_observations``,
        and ``small_sample`` (bool).
    """
    labels = pd.Series(list(labels))
    rows = [
        {
            "regime": regime,
            "n_observations": int((labels == regime).sum()),
            "small_sample": int((labels == regime).sum()) < small_sample_threshold,
        }
        for regime in pd.unique(labels)
    ]
    return pd.DataFrame(rows)
