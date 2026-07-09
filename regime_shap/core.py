"""RegimeSHAPAnalyzer: the one-call entry point for regime-aware SHAP analysis.

Given a fitted tree model, a feature matrix, and a regime specification, the
analyzer runs the whole pipeline: resolve the regimes into per-row labels
(``breaks``), compute global and per-regime SHAP importance and rankings
(``compare``), quantify cross-regime stability with Spearman correlation, the
Akoglu (2018) bands, and bootstrap confidence intervals (``stability``), and
serve figures (``plots``) and reports (``report``). TreeSHAP is computed once in
the constructor and every downstream view is derived from it, except the
bootstrap, which by design recomputes SHAP on resampled rows.
"""

from __future__ import annotations

from collections.abc import Set
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure

from regime_shap import breaks, compare, plots, report, stability


class RegimeSHAPAnalyzer:
    """Regime-aware SHAP feature-importance stability analysis for a tree model.

    Supported models are tree-based (XGBoost, LightGBM, or scikit-learn tree
    ensembles), computed through ``shap.TreeExplainer``. This is deliberate:
    TreeExplainer is exact and fast, which is what makes the bootstrap
    confidence intervals feasible, since they recompute SHAP many times per
    regime. The stability methodology itself is model-agnostic (everything
    downstream operates on SHAP arrays, not the model), so support for other
    model types via pluggable explainers is a possible future extension, with
    the caveat that the bootstrap becomes expensive and approximate for
    non-tree models.
    """

    def __init__(
        self,
        model: object,
        X: pd.DataFrame,
        regimes: object,
        *,
        date_column: str | None = None,
        small_sample_threshold: int = compare.DEFAULT_SMALL_SAMPLE_THRESHOLD,
        moderate_threshold: float = stability.DEFAULT_MODERATE_THRESHOLD,
        stable_threshold: float = stability.DEFAULT_STABLE_THRESHOLD,
        random_state: int = stability.DEFAULT_RANDOM_STATE,
    ) -> None:
        """Build the analyzer and compute global SHAP values once.

        Args:
            model: A fitted tree model compatible with ``shap.TreeExplainer``
                (XGBoost, LightGBM, or a scikit-learn tree model).
            X: The feature matrix the model was trained on (or a comparable set).
            regimes: A regime specification accepted by
                ``breaks.regimes_to_labels`` (per-row labels, date-range tuples,
                or a label-to-range mapping).
            date_column: A datetime column name for the date-range regime forms.
            small_sample_threshold: Regimes below this size are flagged small.
            moderate_threshold: Upper edge of the unstable stability band.
            stable_threshold: Upper edge of the moderate stability band.
            random_state: Seed for the bootstrap, so results are reproducible.

        Raises:
            ValueError: If ``X`` is empty.
        """
        if len(X) == 0:
            raise ValueError("X is empty; provide at least one row")
        self.model = model
        self.X = X
        self.labels = breaks.regimes_to_labels(X, regimes, date_column=date_column)
        self.small_sample_threshold = small_sample_threshold
        self.moderate_threshold = moderate_threshold
        self.stable_threshold = stable_threshold
        self.random_state = random_state
        self._feature_names = list(X.columns)
        self._shap_values = np.asarray(shap.TreeExplainer(model)(X).values)

    def global_importance(self) -> pd.Series:
        """Global mean absolute SHAP per feature, most important first."""
        mean_abs = np.abs(self._shap_values).mean(axis=0)
        importance = pd.Series(mean_abs, index=self._feature_names, name="mean_abs_shap")
        return importance.sort_values(ascending=False)

    def per_regime_importance(self) -> pd.DataFrame:
        """Feature-by-regime mean absolute SHAP."""
        return compare.per_regime_importance(self._shap_values, self._feature_names, self.labels)

    def per_regime_rankings(self) -> pd.DataFrame:
        """Feature-by-regime importance rankings (1 = most important)."""
        return compare.per_regime_rankings(self.per_regime_importance())

    def sample_sizes(self) -> pd.DataFrame:
        """Per-regime observation counts with the small-sample flag."""
        return compare.regime_sample_sizes(
            self.labels, small_sample_threshold=self.small_sample_threshold
        )

    def small_regimes(self) -> Set[str]:
        """The set of regime labels flagged small-sample."""
        sizes = self.sample_sizes()
        return set(sizes.loc[sizes["small_sample"], "regime"])

    def stability_matrix(self) -> pd.DataFrame:
        """Regime-by-regime Spearman correlation of the feature rankings."""
        return stability.pairwise_spearman_matrix(self.per_regime_rankings())

    def stability_classified(self) -> pd.DataFrame:
        """Every regime pair with its Spearman rho, Akoglu band, and small-regime flag."""
        matrix = self.stability_matrix()
        small = self.small_regimes()
        rows = []
        for a, b in combinations(matrix.columns, 2):
            rho = float(matrix.loc[a, b])
            rows.append(
                {
                    "regime_a": a,
                    "regime_b": b,
                    "spearman_rho": rho,
                    "band": stability.classify_stability(
                        rho,
                        moderate_threshold=self.moderate_threshold,
                        stable_threshold=self.stable_threshold,
                    ),
                    "involves_small_regime": a in small or b in small,
                }
            )
        return pd.DataFrame(rows)

    def bootstrap_cis(self, *, n_bootstrap: int = stability.DEFAULT_N_BOOTSTRAP) -> pd.DataFrame:
        """Bootstrap confidence intervals for every regime pair involving a small regime.

        Args:
            n_bootstrap: Number of bootstrap resamples per regime.

        Returns:
            A DataFrame of ``regime_a``, ``regime_b``, ``ci_lower``, ``ci_upper``,
            and ``effective_n`` (the smaller regime's size), one row per pair that
            involves a small-sample regime. Empty if no regime is small.
        """
        small = self.small_regimes()
        sizes = self.sample_sizes().set_index("regime")["n_observations"].to_dict()
        cache: dict[str, np.ndarray] = {}

        def _boot(regime: str) -> np.ndarray:
            if regime not in cache:
                x_regime = self.X.loc[(self.labels == regime).to_numpy()]
                cache[regime] = stability.bootstrap_rankings(
                    self.model, x_regime, n_bootstrap=n_bootstrap, random_state=self.random_state
                )
            return cache[regime]

        rows = []
        for a, b in combinations(self.per_regime_rankings().columns, 2):
            if a in small or b in small:
                lower, upper = stability.bootstrap_spearman_ci(
                    _boot(a), _boot(b), n_bootstrap=n_bootstrap
                )
                rows.append(
                    {
                        "regime_a": a,
                        "regime_b": b,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "effective_n": min(sizes[a], sizes[b]),
                    }
                )
        return pd.DataFrame(rows)

    def plot_global(self, **kwargs) -> Figure:
        """Global importance bar chart (see ``plots.plot_global_importance``)."""
        return plots.plot_global_importance(self.global_importance(), **kwargs)

    def plot_per_regime(self, **kwargs) -> Figure:
        """Per-regime importance heatmap (see ``plots.plot_per_regime_importance``)."""
        return plots.plot_per_regime_importance(self.per_regime_importance(), **kwargs)

    def plot_stability(self, **kwargs) -> Figure:
        """Stability heatmap, with the small regimes flagged and the analyzer's bands."""
        return plots.plot_stability_heatmap(
            self.stability_matrix(),
            small_regimes=self.small_regimes(),
            moderate_threshold=self.moderate_threshold,
            stable_threshold=self.stable_threshold,
            **kwargs,
        )

    def results(self) -> dict[str, pd.DataFrame | pd.Series]:
        """The result tables, ready for ``report`` (HTML, dict, or CSV)."""
        return {
            "global_importance": self.global_importance(),
            "per_regime_importance": self.per_regime_importance(),
            "per_regime_rankings": self.per_regime_rankings(),
            "stability_matrix": self.stability_matrix(),
            "stability_classified": self.stability_classified(),
            "sample_sizes": self.sample_sizes(),
        }

    def to_html(self, **kwargs) -> str:
        """Render the results as a self-contained HTML report."""
        return report.to_html(self.results(), **kwargs)

    def save_csv(self, output_dir: str | Path) -> list[Path]:
        """Write the result tables to CSV files in ``output_dir``."""
        return report.save_csv(self.results(), output_dir)
