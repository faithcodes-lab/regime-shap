# Installation

Install the latest release from PyPI:

```bash
pip install regime-shap
```

## Optional extras

The core install covers the analysis. The optional extras add features you may not need:

- `detection`: the `ruptures` change-point library, needed only for `detect_breaks`.
- `examples`: the tools to run the example notebooks (Jupyter, XGBoost, and others).
- `docs`: the toolchain to build this documentation.
- `dev`: the test and linting toolchain.

Install an extra with, for example:

```bash
pip install "regime-shap[detection]"
```

## Install from source

For development or the latest unreleased changes, install from the repository:

```bash
git clone https://github.com/faithcodes-lab/regime-shap.git
cd regime-shap
pip install -e ".[dev]"
```

## Requirements

Python 3.10 or newer. The core dependencies (numpy, pandas, scipy, shap, matplotlib) are installed
automatically.

## Versioning and licence

`regime-shap` follows Semantic Versioning; during the 0.x series the public API may change between
releases. It is released under the MIT licence.
