# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package scaffolding via bootstrap-package-repo.sh
- Core API: `RegimeSHAPAnalyzer` class (to be implemented in 07-regime-shap-tool.md)

## [0.1.0] - YYYY-MM-DD

Initial release.

### Added
- `RegimeSHAPAnalyzer` for per-regime SHAP computation
- Spearman rank correlation stability metric
- Bootstrap confidence intervals for small-sample regimes
- Visualisation helpers (stability heatmap, per-regime summaries)
- HTML report generation
- Three example notebooks (UK GDP, stock volatility, energy demand)