# Quickstart

`RegimeSHAPAnalyzer` runs the whole analysis in one object. Give it a fitted tree model, the
feature matrix the model was trained on, and a set of regimes, and it computes the SHAP values
once and then serves every view from them.

```python
from xgboost import XGBRegressor

from regime_shap import RegimeSHAPAnalyzer

# X is your feature matrix (date-indexed here), y is the target
model = XGBRegressor().fit(X, y)

# regimes as contiguous date ranges; per-row labels and detect_breaks also work
regimes = {
    "calm": ("2015-01-01", "2019-12-31"),
    "crisis": ("2020-01-01", "2020-12-31"),
    "recovery": ("2021-01-01", "2024-12-31"),
}

analyzer = RegimeSHAPAnalyzer(model, X, regimes)
```

From that one object you can read every result:

```python
analyzer.global_importance()      # mean absolute SHAP per feature, overall
analyzer.per_regime_importance()  # the same, broken down by regime
analyzer.stability_matrix()       # regime-by-regime Spearman correlation of the rankings
analyzer.stability_classified()   # each regime pair with its Akoglu band
```

And produce figures and reports:

```python
analyzer.plot_stability()                       # the banded stability heatmap
analyzer.to_html(title="My stability analysis")  # a self-contained HTML report
analyzer.save_csv("results/")                    # the result tables as CSV files
```

If a regime is small, flag it and read its correlations with bootstrap confidence intervals:

```python
analyzer = RegimeSHAPAnalyzer(model, X, regimes, small_sample_threshold=100)
analyzer.bootstrap_cis()  # intervals for every pair that involves a small regime
```

For the ideas behind these numbers, see [Concepts](concepts.md). For complete worked runs on real
public data, see [Examples](examples.md).
