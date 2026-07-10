# Examples

Two worked notebooks run the whole method on genuinely different public datasets, so the approach
is shown beyond any single domain. Both are in the
[`examples/`](https://github.com/faithcodes-lab/regime-shap/tree/main/examples) directory, and each
bundles a small public data snapshot so it runs offline and reproduces the same figures.

## Finance: market volatility regimes

[`01_finance_market_regimes.ipynb`](https://github.com/faithcodes-lab/regime-shap/blob/main/examples/01_finance_market_regimes.ipynb)

A model for next-day change in the VIX (the equity volatility index), built from public FRED
series. The regimes are found from the data with `detect_breaks`. The result is instability: the
model's drivers reorganise in the 2008 to 2009 crisis, which lands in the unstable band. This shows
the automatic regime-detection path and a case where the drivers shift.

## Energy: electricity demand eras

[`02_energy_demand_regimes.ipynb`](https://github.com/faithcodes-lab/regime-shap/blob/main/examples/02_energy_demand_regimes.ipynb)

A day-ahead model for Great Britain electricity demand, built from public NESO data. The regimes
are hand-labelled as contiguous periods of UK demand history (the higher-demand early 2010s, the
efficiency-driven decline, the 2020 lockdown, and the post-lockdown plateau). The result is
stability: the drivers hold across every era, and the short lockdown era is read with bootstrap
confidence intervals. This shows the hand-labelled regime path and a case where the drivers hold.

## Running them

Install the examples extra and open the notebooks:

```bash
pip install -e ".[examples]"
jupyter notebook
```
