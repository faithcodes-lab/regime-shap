# regime-shap

> Quantify SHAP feature importance stability across structural breaks in time-series data.

[![PyPI version](https://img.shields.io/pypi/v/regime-shap)](https://pypi.org/project/regime-shap/)
[![Documentation Status](https://readthedocs.org/projects/regime-shap/badge/?version=latest)](https://regime-shap.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`regime-shap` extends SHAP feature importance analysis to time-series with structural breaks. Given a pre-trained tree model and a list of regime boundaries, it quantifies how feature importance rankings change across regimes — surfacing instability that may matter for trustworthy interpretation.

## Quick Example

```python
import pandas as pd
from regime_shap import RegimeSHAPAnalyzer

# Assume: X is your feature DataFrame, model is a pre-trained XGBoost/LightGBM model
# regimes is a dict mapping regime labels to (start_date, end_date) tuples

analyzer = RegimeSHAPAnalyzer(model=model, X=X, regimes=regimes)
analyzer.compute()
stability_matrix = analyzer.stability_matrix()  # 6x6 Spearman rho
analyzer.plot_stability_heatmap()
```

## Installation

```bash
pip install regime-shap
```

## Documentation

Full documentation at https://regime-shap.readthedocs.io/

## Origin

This package originated as part of an MSc dissertation at UWE Bristol investigating SHAP stability across UK economic regimes (Brexit, COVID-19). The methodology was extracted into a standalone tool for general use across domains including macroeconomics, finance, and energy.

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

MIT — see `LICENSE` for details.