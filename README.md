# regime-shap

> Quantify SHAP feature importance stability across structural breaks in time-series data.

[![CI](https://github.com/faithcodes-lab/regime-shap/actions/workflows/ci.yml/badge.svg)](https://github.com/faithcodes-lab/regime-shap/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://faithcodes-lab.github.io/regime-shap/)

**Status: pre-release, in active development. Not yet published to PyPI, and the public API is not final.**

Documentation: <https://faithcodes-lab.github.io/regime-shap/>

`regime-shap` extends SHAP feature importance analysis to time-series with structural breaks. Given a pre-trained tree model, a feature matrix, and a set of regimes, it quantifies how the model's feature importance rankings change across regimes, surfacing instability that may matter for trustworthy interpretation.

## Development status

Built so far:
- `breaks`: turn regime specifications into per-row regime labels (plus optional break detection).
- `compare`: per-regime SHAP feature importance and rankings, with small-sample flagging.
- `stability`: pairwise Spearman stability matrix, Akoglu (2018) bands, and bootstrap confidence intervals.
- `plots`: global and per-regime importance figures, and the banded stability heatmap.
- `report`: self-contained HTML report plus dict and CSV export.
- `RegimeSHAPAnalyzer`: a single high-level entry point that ties the modules together.

## Usage

```python
from regime_shap import RegimeSHAPAnalyzer

# model is a fitted tree model, X is the feature matrix, regimes is one label per row
analyzer = RegimeSHAPAnalyzer(model, X, regimes)

analyzer.stability_matrix()      # regime-by-regime Spearman correlation of importance rankings
analyzer.stability_classified()  # each regime pair with its Akoglu (2018) stability band
analyzer.plot_stability()        # the banded stability heatmap
analyzer.to_html(title="...")    # a self-contained report of every result table
```

The building-block functions in `regime_shap.breaks`, `regime_shap.compare`, and `regime_shap.stability` are also available directly if you want finer control.

## Supported models

`regime-shap` currently supports tree-based models (for example XGBoost, LightGBM, and scikit-learn tree ensembles) through SHAP's `TreeExplainer`. This is a deliberate choice: `TreeExplainer` computes exact SHAP values quickly, which is what makes the bootstrap confidence intervals feasible, since they recompute SHAP up to a thousand times per regime.

The stability methodology itself is model-agnostic. The comparison, stability, plotting, and report steps operate on SHAP values alone and never inspect the model, so only the SHAP computation is tree-specific.

Support for other model types through pluggable explainers (for example `KernelExplainer` for arbitrary models, or `LinearExplainer` for linear ones) is a possible future extension, tracked in [issue #8](https://github.com/faithcodes-lab/regime-shap/issues/8). The honest caveat is that the bootstrap confidence intervals become expensive and approximate for non-tree models, because general explainers are slower and introduce sampling variance.

## Installation

Not yet on PyPI. For development, install from source:

```bash
git clone https://github.com/faithcodes-lab/regime-shap.git
cd regime-shap
pip install -e ".[dev]"
```

## Origin

This package generalises a SHAP stability methodology, first developed for analysing UK economic regimes (the Global Financial Crisis, Brexit, and COVID-19), into a standalone, domain-agnostic tool. Its examples span finance and energy as well as macroeconomics.

If you use `regime-shap` in research, please cite:

```bibtex
@software{olan_george_regime_shap,
  author = {Olan-George, Faith},
  title = {regime-shap: Quantifying SHAP Feature Importance Stability Across Structural Breaks},
  year = {2026},
  url = {https://github.com/faithcodes-lab/regime-shap},
}
```

## Licence

MIT, see `LICENSE` for details.
