"""Cross-regime stability of SHAP feature-importance rankings.

The question this module answers: do the features the model leans on stay the
same across regimes, or do they reshuffle? Two regimes' feature rankings are
compared with Spearman rank correlation (rho): 1 means identical orderings, -1
reversed, near 0 unrelated. ``pairwise_spearman_matrix`` builds the full
regime-by-regime table, and ``classify_stability`` labels a rho using the
Akoglu (2018) bands.

Small regimes give noisy rankings, so ``bootstrap_rankings`` resamples a
regime's rows with replacement and recomputes the SHAP rankings each time, and
``bootstrap_spearman_ci`` turns those into a confidence interval on rho, the
honest uncertainty a handful of observations carries rather than a single
falsely precise number. Rankings, not raw SHAP magnitudes, are compared, so the
result is not confounded by volatility differences between regimes.

Akoglu, H. (2018) 'User's guide to correlation coefficients', Turkish Journal
of Emergency Medicine, 18(3), pp. 91-93.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

# Akoglu (2018) bands: unstable when rho <= 0.3, moderately stable when
# 0.3 < rho <= 0.6, stable when rho > 0.6.
DEFAULT_MODERATE_THRESHOLD = 0.3
DEFAULT_STABLE_THRESHOLD = 0.6
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_RANDOM_STATE = 42
# Below this many rows a bootstrap CI is too unreliable to trust; warn but proceed.
MIN_BOOTSTRAP_OBS = 4


def pairwise_spearman_matrix(rankings: pd.DataFrame) -> pd.DataFrame:
    """Regime-by-regime Spearman correlation of the per-regime feature rankings.

    Args:
        rankings: A feature-by-regime DataFrame of ranks, as returned by
            ``compare.per_regime_rankings``.

    Returns:
        A square regime-by-regime DataFrame of Spearman rho, symmetric with a
        unit diagonal.
    """
    regimes = list(rankings.columns)
    matrix = pd.DataFrame(index=regimes, columns=regimes, dtype=float)
    for a in regimes:
        for b in regimes:
            rho, _ = spearmanr(rankings[a], rankings[b])
            matrix.loc[a, b] = rho
    return matrix


def classify_stability(
    rho: float,
    *,
    moderate_threshold: float = DEFAULT_MODERATE_THRESHOLD,
    stable_threshold: float = DEFAULT_STABLE_THRESHOLD,
) -> str:
    """Label a Spearman rho by the Akoglu (2018) bands.

    Args:
        rho: A Spearman rank correlation.
        moderate_threshold: Upper edge of the unstable band (default 0.3).
        stable_threshold: Upper edge of the moderate band (default 0.6).

    Returns:
        "stable" when rho > stable_threshold, "moderately stable" when
        moderate_threshold < rho <= stable_threshold, else "unstable".
    """
    if rho > stable_threshold:
        return "stable"
    if rho > moderate_threshold:
        return "moderately stable"
    return "unstable"


def bootstrap_rankings(
    model: object,
    X_regime: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Bootstrap the SHAP feature rankings within one regime.

    Resamples the regime's rows with replacement ``n_bootstrap`` times, and for
    each resample recomputes TreeSHAP and ranks the features by mean absolute
    SHAP. Exploratory: this quantifies the uncertainty a small regime's ranking
    already has, it does not remove it.

    Args:
        model: A fitted tree model compatible with ``shap.TreeExplainer``.
        X_regime: The feature rows belonging to one regime.
        n_bootstrap: Number of resamples.
        random_state: Seed, so the result is reproducible.

    Returns:
        An ``(n_bootstrap, n_features)`` array of rankings, one row per resample.

    Warns:
        If the regime has fewer than ``MIN_BOOTSTRAP_OBS`` rows, the CI derived
        from these rankings will be unreliable.
    """
    n = len(X_regime)
    if n < MIN_BOOTSTRAP_OBS:
        warnings.warn(
            f"regime has {n} observations (< {MIN_BOOTSTRAP_OBS}); "
            "bootstrap confidence intervals will be unreliable",
            stacklevel=2,
        )
    rng = np.random.default_rng(random_state)
    explainer = shap.TreeExplainer(model)
    n_features = X_regime.shape[1]
    out = np.empty((n_bootstrap, n_features))
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        explanation = explainer(X_regime.iloc[idx])
        mean_abs = np.abs(explanation.values).mean(axis=0)
        out[i] = pd.Series(mean_abs).rank(ascending=False, method="average").to_numpy()
    return out


def bootstrap_spearman_ci(
    rankings_a: np.ndarray,
    rankings_b: np.ndarray,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Confidence interval for Spearman rho between two regimes' bootstrap rankings.

    Pairs resample ``i`` of one regime with resample ``i`` of the other, computes
    rho for each pair, and returns the requested percentile interval of that
    distribution. For a pair involving a small regime this interval is often
    wide, which honestly reflects the uncertainty already present.

    Args:
        rankings_a: An ``(n_bootstrap, n_features)`` array from ``bootstrap_rankings``.
        rankings_b: The other regime's array of the same shape.
        n_bootstrap: Cap on the number of paired draws used.
        ci: Confidence level (default 0.95 for a 95% interval).

    Returns:
        A ``(lower, upper)`` tuple of rho at the interval's edges.
    """
    n = min(len(rankings_a), len(rankings_b), n_bootstrap)
    rhos = np.array([spearmanr(rankings_a[i], rankings_b[i])[0] for i in range(n)])
    lower_pct = (1.0 - ci) / 2.0 * 100.0
    upper_pct = (1.0 + ci) / 2.0 * 100.0
    return float(np.percentile(rhos, lower_pct)), float(np.percentile(rhos, upper_pct))
