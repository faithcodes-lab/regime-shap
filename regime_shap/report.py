"""Assemble the regime-aware SHAP results into shareable reports.

The analysis produces a handful of tables (global importance, per-regime
rankings, the stability matrix and its band classifications, sample sizes, and
any bootstrap confidence intervals). This module turns that collection into a
self-contained HTML report as the headline output, and exposes the same tables
as a plain dictionary and as CSV files so the results stay machine-readable.
Callers pass a mapping of ``{name: table}``; the module does not assume any
particular set of tables, so it stays general.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

Tables = Mapping[str, "pd.DataFrame | pd.Series"]

_STYLE = (
    "body{font-family:sans-serif;margin:2rem;color:#222;}"
    "h1{font-size:1.6rem;}h2{font-size:1.2rem;margin-top:1.5rem;}"
    "table{border-collapse:collapse;margin-top:0.5rem;}"
    "th,td{border:1px solid #ccc;padding:4px 10px;text-align:right;}"
    "th{background:#f4f4f4;}"
    ".notes{color:#555;}"
)


def to_dict(tables: Tables) -> dict:
    """Return the result tables as JSON-serialisable nested dicts.

    Args:
        tables: A mapping of table name to DataFrame or Series.

    Returns:
        A dict keyed by table name; each DataFrame becomes a
        ``{column: {index: value}}`` dict and each Series an ``{index: value}``
        dict.
    """
    return {name: table.to_dict() for name, table in tables.items()}


def save_csv(tables: Tables, output_dir: str | Path) -> list[Path]:
    """Write each table to ``output_dir/<name>.csv``.

    Args:
        tables: A mapping of table name to DataFrame or Series.
        output_dir: Directory to write into; created if it does not exist.

    Returns:
        The list of written file paths, in the order of ``tables``.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path)
        paths.append(path)
    return paths


def to_html(
    tables: Tables,
    *,
    title: str = "regime-shap stability report",
    notes: list[str] | None = None,
) -> str:
    """Render the tables as a single self-contained HTML report.

    Args:
        tables: A mapping of table name to DataFrame or Series, rendered in
            order under headings.
        title: The page and top heading title.
        notes: Optional caveats shown near the top, for example that small
            regimes should be read with their bootstrap confidence intervals.

    Returns:
        A complete HTML document as a string.
    """
    parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{title}</title>",
        f"<style>{_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
    ]
    if notes:
        parts.append("<ul class='notes'>")
        parts.extend(f"<li>{note}</li>" for note in notes)
        parts.append("</ul>")
    for name, table in tables.items():
        frame = table.to_frame() if isinstance(table, pd.Series) else table
        parts.append(f"<h2>{name}</h2>")
        parts.append(frame.to_html(border=0, float_format=lambda x: f"{x:.4f}"))
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)
