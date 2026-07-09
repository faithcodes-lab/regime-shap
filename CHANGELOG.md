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

### Planned
- `RegimeSHAPAnalyzer` entry point, plotting, and HTML/CSV report modules.
- Example notebooks (macroeconomic, financial, energy).
- Documentation site and the first PyPI release (v0.1.0).
