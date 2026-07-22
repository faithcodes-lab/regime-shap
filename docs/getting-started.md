# Getting started

The fastest ways to go from zero to a stability heatmap. Pick whichever suits you.

## 1. Try it in your browser (no install)

Open the live demo and explore two built-in examples:
[regime-shap demo on Hugging Face Spaces](https://huggingface.co/spaces/FaithCodes/regime-shap-demo).
Adjust the stability thresholds and read the regime-by-regime heatmap without installing anything.

## 2. Install and run on your own data

```bash
pip install "regime-shap[examples]"
```

The `[examples]` extra adds Jupyter and XGBoost so you can run the notebooks. Keep the double
quotes; some shells treat `[examples]` as a pattern otherwise. The same command works on macOS,
Linux, and Windows (PowerShell or Command Prompt).

The whole analysis runs from one object: give it a fitted tree model, the feature matrix, and a
set of regimes (one label per row, or date ranges).

```python
from xgboost import XGBRegressor
from regime_shap import RegimeSHAPAnalyzer

model = XGBRegressor().fit(X, y)         # your fitted tree model
regimes = df["your_period_column"]        # one label per row (a season, era, or period)

analyzer = RegimeSHAPAnalyzer(model, X, regimes)
analyzer.stability_classified()           # each regime pair, with its stability band
analyzer.plot_stability()                 # the banded heatmap
analyzer.to_html(title="My analysis")     # a self-contained HTML report
```

See the [Quickstart](quickstart.md) for the full API, and [Concepts](concepts.md) for what the
numbers mean.

## 3. No data of your own? Use the practice pack

Download `practice-datasets.zip` from the
[latest release](https://github.com/faithcodes-lab/regime-shap/releases/latest). It contains ten
ready-to-use public datasets, a README with a copy-paste settings block for each, and a
fill-in-the-blanks notebook, `try_your_own_data.ipynb`.

Open the notebook and run it: it works on a bundled dataset out of the box, so you see a full
result straight away. Then edit the one config cell to point at another dataset, or at your own
CSV.

## Tell us how it went

Feedback of any kind is welcome, including what confused you. Post in
[GitHub Discussions](https://github.com/faithcodes-lab/regime-shap/discussions).

One caveat worth knowing up front: `regime-shap` measures how stable a model's explanations are,
not whether the model itself is any good. It will report "stable" even for a weak or leaky model,
so read the stability result alongside your model's own performance.
