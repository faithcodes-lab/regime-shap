# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The package is in active development. Nothing has been released yet.

### Added
- Package scaffolding and CI (via bootstrap-package-repo.sh).
- `breaks`: resolve regime specifications (per-row labels, date-range tuples, or a label-to-range mapping) into per-row labels, with optional ruptures-based break detection.
- `compare`: per-regime SHAP feature importance and rankings, with per-regime sample sizes and a small-sample flag.
- `stability`: pairwise Spearman rank-correlation matrix, the Akoglu (2018) stability bands, and bootstrap confidence intervals for small regimes.
- `plots`: global importance bar, per-regime importance heatmap, and the Akoglu-banded stability heatmap with small-regime flags.
- `report`: self-contained HTML report plus dict and CSV export of the result tables.
- `RegimeSHAPAnalyzer`: the one-call entry point that ties the modules together, exported from the package top level.

### Documentation
- Documented the supported-model boundary: the package is tree-only via `TreeExplainer`, the stability methodology is model-agnostic, and pluggable explainers for other models are a possible future extension (tracked in issue #8).

### Planned
- Example notebooks (macroeconomic, financial, energy).
- Documentation site and the first PyPI release (v0.1.0).
