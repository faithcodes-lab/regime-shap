---
title: Regime SHAP Demo
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.20.0"
app_file: app.py
pinned: false
license: mit
short_description: Interactive demo of regime-aware SHAP stability analysis
---

# regime-shap demo

Interactive demonstration of [`regime-shap`](https://github.com/faithcodes-lab/regime-shap), a
Python package that measures how stable a tree model's SHAP feature importance is across regimes
(distinct time periods such as structural breaks).

Pick one of the two built-in examples, adjust the stability thresholds, and read the
regime-by-regime stability heatmap, the banded classification, and the global feature importance.

- **Finance**: a model for the next-day change in market volatility (the VIX), with regimes found
  by break detection. Its drivers shift in the 2008 to 2009 crisis.
- **Energy**: a day-ahead model for Great Britain electricity demand, with hand-labelled eras of
  UK demand history. Its drivers hold across the eras.

## Links

- Source code: https://github.com/faithcodes-lab/regime-shap
- Documentation: https://faithcodes-lab.github.io/regime-shap/

The package is pre-release and not yet on PyPI, so this Space installs it from source.
