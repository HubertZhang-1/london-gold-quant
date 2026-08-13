# -*- coding: utf-8 -*-
"""Intraday strategies for gold, based on the UTC calendar day."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import sma


def open_range_breakout_signals(
    df: pd.DataFrame,
    range_hours: int = 3,
    ma_filter: int = 0,
    stop_mult: float = 1.0,
) -> pd.DataFrame:
    """Daily open-range breakout: trade breaks of the first N hourly bars.

    Signals are generated at bar close and executed on the next bar open.
    Positions are flattened on the last bar of the UTC day.
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)
    days = out["date"].dt.date.to_numpy()
    closes = out["close"].to_numpy(dtype=float)
    highs = out["high"].to_numpy(dtype=float)
    lows = out["low"].to_numpy(dtype=float)
    ma = sma(out["close"], ma_filter) if ma_filter > 0 else None

    signals = np.zeros(n, dtype=int)
    stops = np.zeros(n)
    range_highs = np.full(n, np.nan)
    range_lows = np.full(n, np.nan)

    i = 0
    while i < n:
        day = days[i]
        j = i
        while j < n and days[j] == day:
            j += 1

        if j - i >= range_hours + 1:
            rh = float(np.max(highs[i : i + range_hours]))
            rl = float(np.min(lows[i : i + range_hours]))
            width = rh - rl
            pos = 0
            for k in range(i + range_hours, j):
                c = closes[k]
                if pos == 0:
                    trend_ok = ma_filter <= 0 or not np.isnan(ma.iloc[k])
                    if trend_ok and c > rh and (ma_filter <= 0 or c > ma.iloc[k]):
                        signals[k] = 1
                        pos = 1
                        stops[k] = width * stop_mult if width > 0 else 0.0
                    elif trend_ok and c < rl and (ma_filter <= 0 or c < ma.iloc[k]):
                        signals[k] = -1
                        pos = -1
                        stops[k] = width * stop_mult if width > 0 else 0.0
                elif pos > 0:
                    if k == j - 1:
                        signals[k] = 0
                        pos = 0
                    else:
                        signals[k] = 1
                else:
                    if k == j - 1:
                        signals[k] = 0
                        pos = 0
                    else:
                        signals[k] = -1

            range_highs[i : min(i + range_hours, j)] = rh
            range_lows[i : min(i + range_hours, j)] = rl
        i = j

    out["signal"] = signals
    out["stop_dist"] = stops
    out["range_high"] = range_highs
    out["range_low"] = range_lows
    out["ma"] = ma
    return out
