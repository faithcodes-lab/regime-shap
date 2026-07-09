"""regime-shap: Quantify SHAP feature importance stability across structural breaks.

Public API:
- RegimeSHAPAnalyzer: the one-call entry point for regime-aware SHAP analysis.
"""

from regime_shap.core import RegimeSHAPAnalyzer

__version__ = "0.1.0"

__all__ = ["RegimeSHAPAnalyzer"]
