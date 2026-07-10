# Installation

`regime-shap` is in active development and has not been released to PyPI yet, so for now install
it from source.

```bash
git clone https://github.com/faithcodes-lab/regime-shap.git
cd regime-shap
pip install -e .
```

## Optional extras

The core install covers the analysis. The optional extras add features you may not need:

- `detection`: the `ruptures` change-point library, needed only for `detect_breaks`.
- `examples`: the tools to run the example notebooks (Jupyter, XGBoost, and others).
- `docs`: the toolchain to build this documentation.
- `dev`: the test and linting toolchain.

Install an extra with, for example:

```bash
pip install -e ".[detection]"
```

## Requirements

Python 3.10 or newer. The core dependencies (numpy, pandas, scipy, shap, matplotlib) are installed
automatically.

A PyPI release is planned, after which `pip install regime-shap` will work directly.
