# -*- coding: utf-8 -*-
"""Markdown and SVG reporting helpers (no third-party plotting dependency)."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

PALETTE = ["#d97706", "#2563eb", "#059669", "#dc2626", "#7c3aed", "#0891b2"]


def write_equity_svg(results, path: str | Path, width: int = 980, height: int = 460) -> None:
    """Draw one normalized equity line per (label, dates, equity) tuple."""
    labels = [r[0] for r in results]
    series = []
    for _, _, equity in results:
        values = pd.Series(equity).ffill().to_numpy()
        series.append(values)

    margin = {"l": 70, "r": 24, "t": 26, "b": 56}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]

    all_v = np.concatenate(series) if series else np.array([1.0])
    y_min = float(np.nanmin(all_v))
    y_max = float(np.nanmax(all_v))
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.06
    y_min -= pad
    y_max += pad

    max_len = max((len(s) for s in series), default=1)
    lines = []
    for idx, (label, values) in enumerate(zip(labels, series)):
        points = []
        for i, v in enumerate(values):
            if np.isnan(v):
                continue
            x = margin["l"] + (i / max(1, max_len - 1)) * plot_w
            y = margin["t"] + (1 - (v - y_min) / (y_max - y_min)) * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            color = PALETTE[idx % len(PALETTE)]
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                f'points="{" ".join(points)}" />'
            )
            lines.append(
                f'<text x="{margin["l"] + 8}" y="{margin["t"] + 18 + idx * 20}" '
                f'fill="{color}" font-size="13">{escape(label)}</text>'
            )

    grid = []
    for i in range(5):
        ratio = i / 4.0
        y = margin["t"] + ratio * plot_h
        value = y_max - (y_max - y_min) * ratio
        grid.append(
            f'<line x1="{margin["l"]}" y1="{y:.1f}" x2="{width - margin["r"]}" '
            f'y2="{y:.1f}" stroke="#d1d5db" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{margin["l"] - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#374151" font-size="12">{value:,.0f}</text>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<text x="{width / 2:.0f}" y="18" text-anchor="middle" font-size="15" '
        'fill="#111827">London Gold Equity Curves (normalized)</text>',
        *grid,
        *lines,
        '<line x1="' + str(margin["l"]) + f'" y1="{margin["t"]}" x2="{margin["l"]}" '
        f'y2="{margin["t"] + plot_h}" stroke="#374151" stroke-width="1.5" />',
        '<line x1="' + str(margin["l"]) + f'" y1="{margin["t"] + plot_h}" x2="{width - margin["r"]}" '
        f'y2="{margin["t"] + plot_h}" stroke="#374151" stroke-width="1.5" />',
        "</svg>",
    ]
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def stats_row(stats: dict, name: str) -> list:
    return [
        name,
        stats["total_return"],
        stats["annual_return"],
        stats["sharpe"],
        stats["max_drawdown"],
        stats["trade_count"],
        stats["win_rate"],
        stats["profit_factor"],
    ]


def format_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [str(v) if not isinstance(v, float) else f"{v:.2f}" for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
