"""Hugging Face Spaces demo for regime-shap.

Two preset examples (finance and energy) run the whole package pipeline and show
the stability heatmap, the banded classification, per-regime importance, and
global importance. Bands use the package's default thresholds (Akoglu, 2018).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render figures without a display

import numpy as np
import pandas as pd
import gradio as gr
from xgboost import XGBRegressor

from regime_shap import RegimeSHAPAnalyzer

DATA = Path(__file__).parent / "data"

FINANCE = "Finance: next-day market volatility (VIX)"
ENERGY = "Energy: GB electricity demand"

_XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)


def _finance_analyzer() -> RegimeSHAPAnalyzer:
    raw = pd.read_csv(DATA / "market_regimes.csv", parse_dates=["date"]).set_index("date")
    df = pd.DataFrame(index=raw.index)
    df["vix"] = raw["vix"]
    df["vix_ma5"] = raw["vix"].rolling(5).mean()
    df["vix_ma20"] = raw["vix"].rolling(20).mean()
    df["vix_mom5"] = raw["vix"] - raw["vix"].shift(5)
    df["term_spread"] = raw["term_spread"]
    df["dgs10"] = raw["dgs10"]
    df["oil_chg5"] = raw["oil"].diff(5)
    df["oil_vol20"] = raw["oil"].diff().rolling(20).std()
    target = (raw["vix"].shift(-1) - raw["vix"]).rename("y")
    data = df.join(target).dropna()
    X, y = data.drop(columns="y"), data["y"]

    # These eras were found by detect_breaks on the VIX series (see the finance
    # notebook). They are precomputed here as date ranges so the demo stays
    # responsive, since running detect_breaks on this many points takes ~40s.
    eras = {
        "1990-1996": ("1990-01-31", "1996-12-09"),
        "1996-2008": ("1996-12-10", "2008-09-22"),
        "2008-2009": ("2008-09-23", "2009-04-30"),  # the global financial crisis
        "2009-2026": ("2009-05-01", "2026-07-02"),
    }
    model = XGBRegressor(**_XGB).fit(X, y)
    return RegimeSHAPAnalyzer(model, X, eras)


def _energy_analyzer() -> RegimeSHAPAnalyzer:
    raw = pd.read_csv(DATA / "gb_electricity_demand.csv", parse_dates=["date"]).set_index("date")
    df = pd.DataFrame(index=raw.index)
    df["demand"] = raw["demand_mean"]
    df["demand_lag7"] = raw["demand_mean"].shift(7)
    df["is_weekend"] = (raw.index.dayofweek >= 5).astype(int)
    doy = raw.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["wind"] = raw["wind"]
    df["solar"] = raw["solar"]
    target = raw["demand_mean"].shift(-1).rename("y")
    data = df.join(target).dropna()
    X, y = data.drop(columns="y"), data["y"]

    eras = {
        "2010-2014": ("2010-01-01", "2014-12-31"),
        "2015-2020": ("2015-01-01", "2020-03-23"),
        "2020 lockdown": ("2020-03-24", "2020-12-31"),
        "2021-2025": ("2021-01-01", "2025-12-31"),
    }
    model = XGBRegressor(**_XGB).fit(X, y)
    return RegimeSHAPAnalyzer(model, X, eras, small_sample_threshold=500)


# Build each analyzer (fits the model and computes SHAP) once, then reuse it.
_BUILDERS = {FINANCE: _finance_analyzer, ENERGY: _energy_analyzer}
_CACHE: dict[str, RegimeSHAPAnalyzer] = {}


def _get(dataset: str) -> RegimeSHAPAnalyzer:
    if dataset not in _CACHE:
        _CACHE[dataset] = _BUILDERS[dataset]()
    return _CACHE[dataset]


def analyse(dataset: str):
    an = _get(dataset)

    fig = an.plot_stability()
    per_regime_fig = an.plot_per_regime()
    # SHAP values come out as float32; round after casting to float64, otherwise the
    # float32->float64 widening resurrects long garbage decimals (e.g. 0.1834000051...).
    per_regime_table = (
        an.per_regime_importance()
        .astype("float64")
        .round(4)
        .rename_axis("feature")
        .reset_index()
    )
    global_fig = an.plot_global()
    classified = an.stability_classified().round({"spearman_rho": 3})
    importance = (
        an.global_importance().astype("float64").round(4).rename_axis("feature").reset_index()
    )

    bands = classified["band"].value_counts().to_dict()
    summary = ", ".join(f"{n} {b}" for b, n in bands.items())
    note = (
        f"**{dataset}**\n\n"
        f"Regime pairs by band: {summary}.\n\n"
        "Finance regimes are found automatically with `detect_breaks`; energy regimes are "
        "hand-labelled eras of UK demand history."
    )
    return fig, classified, per_regime_fig, per_regime_table, global_fig, importance, note


_CSS = """
.pair-row { align-items: flex-start !important; }
"""

with gr.Blocks(title="regime-shap demo", css=_CSS) as demo:
    gr.Markdown(
        "# regime-shap\n"
        "Quantify how stable a tree model's SHAP feature importance is across regimes "
        "(distinct time periods such as structural breaks). Pick an example dataset and "
        "run the analysis, it trains a fresh model and computes SHAP live, nothing is "
        "precomputed except the regime boundaries themselves.\n\n"
        "Source: [github.com/faithcodes-lab/regime-shap](https://github.com/faithcodes-lab/regime-shap) "
        "· Docs: [faithcodes-lab.github.io/regime-shap](https://faithcodes-lab.github.io/regime-shap/)"
    )
    with gr.Row():
        dataset = gr.Dropdown([FINANCE, ENERGY], value=FINANCE, label="Example dataset")
        run = gr.Button("Run analysis", variant="primary")

    with gr.Column(visible=False) as results_group:
        gr.Markdown("## Regime-pair stability")
        with gr.Row(elem_classes="pair-row"):
            heatmap = gr.Plot(label="Stability heatmap")
            table = gr.Dataframe(
                label="Regime pairs and stability bands", wrap=True, max_height=340
            )
        gr.Markdown(
            "*Colour shows the stability band: green is stable, orange is moderately "
            "stable, red is unstable, using the Akoglu (2018) thresholds.*"
        )

        gr.Markdown("---\n## Per-regime feature importance")
        with gr.Row(elem_classes="pair-row"):
            per_regime_plot = gr.Plot(
                label="Per-regime SHAP feature importance", show_label=False
            )
            per_regime_table = gr.Dataframe(
                label="Per-regime SHAP feature importance (mean absolute SHAP)",
                wrap=True,
                max_height=340,
            )
        gr.Markdown(
            "*Colour shows the size of each feature's mean absolute SHAP value in "
            "that regime, from dark (low) to yellow (high).*"
        )

        gr.Markdown("---\n## Global feature importance")
        with gr.Row(elem_classes="pair-row"):
            global_plot = gr.Plot(label="Global SHAP feature importance", show_label=False)
            imp = gr.Dataframe(
                label="Global feature importance (mean absolute SHAP)",
                wrap=True,
                max_height=340,
            )

        gr.Markdown("---\n### Interpretation")
        interpretation = gr.Markdown()

    inputs = [dataset]
    outputs = [heatmap, table, per_regime_plot, per_regime_table, global_plot, imp, interpretation]
    run.click(analyse, inputs=inputs, outputs=outputs).then(
        lambda: gr.update(visible=True), outputs=results_group
    )


if __name__ == "__main__":
    demo.launch()
