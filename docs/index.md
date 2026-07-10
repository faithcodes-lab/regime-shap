# regime-shap

Quantify how stable a tree model's SHAP feature importance is across structural breaks in
time series data.

`regime-shap` takes a fitted tree model, a feature matrix, and a set of regimes (distinct
time periods such as a crisis, a policy change, or an economic era), and measures whether the
features the model relies on stay the same across those regimes or reorganise. A model can
score well overall while resting on unstable foundations, and this package makes that visible.

```{note}
The package is in active development and has not been released to PyPI yet. The public API may
still change. See [Installation](installation.md) for installing from source.
```

## Where to start

- [Installation](installation.md): install from source while the package is pre-release.
- [Quickstart](quickstart.md): the one-call `RegimeSHAPAnalyzer` workflow.
- [Concepts](concepts.md): the method in full, regimes, per-regime SHAP importance, Spearman
  stability, the Akoglu bands, and bootstrap confidence intervals.
- [Examples](examples.md): worked notebooks on public finance and energy data.
- [API reference](api.md): the public functions and classes.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
concepts
examples
api
changelog
```
