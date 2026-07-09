# regime-shap

> Quantify SHAP feature importance stability across structural breaks in time-series data.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Status: pre-release, in active development. Not yet published to PyPI, and the public API is not final.**

`regime-shap` extends SHAP feature importance analysis to time-series with structural breaks. Given a pre-trained tree model, a feature matrix, and a set of regimes, it quantifies how the model's feature importance rankings change across regimes, surfacing instability that may matter for trustworthy interpretation.

## Development status

Built so far:
- `breaks`: turn regime specifications into per-row regime labels (plus optional break detection).
- `compare`: per-regime SHAP feature importance and rankings, with small-sample flagging.
- `stability`: pairwise Spearman stability matrix, Akoglu (2018) bands, and bootstrap confidence intervals.

Planned (not yet implemented):
- `RegimeSHAPAnalyzer`, a single high-level entry point.
- Plotting and HTML/CSV report helpers.
- Example notebooks, documentation, and a first PyPI release.

## Planned API

The intended high-level interface, once the analyzer is built, is roughly:

```python
from regime_shap import RegimeSHAPAnalyzer  # not yet implemented

analyzer = RegimeSHAPAnalyzer(model=model, X=X, regimes=regimes)
stability = analyzer.stability_matrix()
analyzer.plot_stability()
```

Until then, the building-block functions in `regime_shap.breaks`, `regime_shap.compare`, and `regime_shap.stability` are available.

## Installation

Not yet on PyPI. For development, install from source:

```bash
git clone https://github.com/faithcodes-lab/regime-shap.git
cd regime-shap
pip install -e ".[dev]"
```

## Origin

This package is being extracted from an MSc dissertation at UWE Bristol investigating SHAP stability across UK economic regimes (the Global Financial Crisis, Brexit, and COVID-19). The methodology is being generalised into a standalone tool for use across domains such as macroeconomics, finance, and energy.

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
