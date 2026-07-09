# Examples

Worked examples showing `regime-shap` across genuinely different public datasets, so the method
is demonstrated beyond any single domain. Each notebook is self-contained and uses public data
bundled under `data/`.

- `01_finance_market_regimes.ipynb` — Finance. An XGBoost model for next-day market volatility
  (the VIX), with regimes found from the data by `detect_breaks`. The drivers **change** in a
  crisis: the stability analysis flags the 2008 to 2009 regime as the unstable outlier.
- `02_energy_seasonal_demand.ipynb` — Energy. A model for next-day change in Great Britain
  electricity demand, with regimes **labelled by hand** as the four seasons plus the 2020
  lockdown. The drivers **hold** across the seasons, and the short lockdown regime is read with
  bootstrap confidence intervals.

Between them the two notebooks cover both ways of supplying regimes (data-driven break detection
and hand labels) and both kinds of outcome a stability analysis can report (drivers that shift
and drivers that hold).

## Running

Install with the examples extra, then launch Jupyter:

```bash
pip install regime-shap[examples]
jupyter notebook
```

Each notebook reads its bundled snapshot from `data/`, so it runs offline and reproduces the
same figures. The data provenance and the exact fetch code are documented in `data/README.md`
and shown in each notebook.
