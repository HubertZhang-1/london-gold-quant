# -*- coding: utf-8 -*-
"""Second-generation intraday gold strategies.

These strategies replace the plain UTC-day open range breakout with
session-aware, momentum and volatility-filtered logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr, sma

SESSION_HOURS = {
    "london": (7, 16),
    "ny": (13, 21),
}


def _atr_ok(atr14: pd.Series, atr_avg: pd.Series, i: int, enabled: bool) -> bool:
    if not enabled:
        return True
    a = atr14.iloc[i]
    b = atr_avg.iloc[i]
    return not np.isnan(a) and not np.isnan(b) and a > b


def session_breakout_signals(
    df: pd.DataFrame,
    session: str = "london",
    range_bars: int = 2,
    stop_mult: float = 1.5,
    ma_filter: int = 24,
    atr_filter: bool = True,
) -> pd.DataFrame:
    """Breakout of the first bars of the London or New York session."""
    out = df.copy().reset_index(drop=True)
    start_hour, end_hour = SESSION_HOURS[session]
    n = len(out)
    hours = out["date"].dt.hour.to_numpy()
    days = out["date"].dt.date.to_numpy()
    closes = out["close"].to_numpy(dtype=float)
    highs = out["high"].to_numpy(dtype=float)
    lows = out["low"].to_numpy(dtype=float)

    atr14 = atr(out["high"], out["low"], out["close"], 14)
    atr_avg = atr14.rolling(7 * 24).mean()
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
        idx = [k for k in range(i, j) if start_hour <= hours[k] < end_hour]
        if len(idx) >= range_bars + 1:
            first_bars = idx[:range_bars]
            rh = float(np.max(highs[first_bars]))
            rl = float(np.min(lows[first_bars]))
            width = rh - rl
            pos = 0
            for k in idx[range_bars:]:
                c = closes[k]
                vol_ok = _atr_ok(atr14, atr_avg, k, atr_filter)
                if pos == 0:
                    trend_ok = ma_filter <= 0 or not np.isnan(ma.iloc[k])
                    if vol_ok and trend_ok and c > rh and (ma_filter <= 0 or c > ma.iloc[k]):
                        signals[k] = 1
                        pos = 1
                        stops[k] = width * stop_mult if width > 0 else 0.0
                    elif vol_ok and trend_ok and c < rl and (ma_filter <= 0 or c < ma.iloc[k]):
                        signals[k] = -1
                        pos = -1
                        stops[k] = width * stop_mult if width > 0 else 0.0
                elif pos > 0:
                    signals[k] = 0 if k == idx[-1] else 1
                    if k == idx[-1]:
                        pos = 0
                else:
                    signals[k] = 0 if k == idx[-1] else -1
                    if k == idx[-1]:
                        pos = 0
            range_highs[first_bars] = rh
            range_lows[first_bars] = rl
        i = j

    out["signal"] = signals
    out["stop_dist"] = stops
    out["range_high"] = range_highs
    out["range_low"] = range_lows
    out["ma"] = ma
    out["atr"] = atr14
    return out


def momentum_trend_signals(
    df: pd.DataFrame,
    fast_bars: int = 6,
    slow_bars: int = 24,
    ma_filter: int = 48,
    stop_mult: float = 2.0,
) -> pd.DataFrame:
    """Dual momentum: short and medium ROC aligned with the trend filter."""
    out = df.copy().reset_index(drop=True)
    close = out["close"]
    fast_roc = close.pct_change(fast_bars)
    slow_roc = close.pct_change(slow_bars)
    ma = sma(close, ma_filter) if ma_filter > 0 else None
    atr14 = atr(out["high"], out["low"], close, 14)
    n = len(out)

    signals = np.zeros(n, dtype=int)
    stops = np.zeros(n)
    for i in range(n):
        fr = fast_roc.iloc[i]
        sr = slow_roc.iloc[i]
        if np.isnan(fr) or np.isnan(sr):
            continue
        trend_ok = ma_filter <= 0 or not np.isnan(ma.iloc[i])
        c = close.iloc[i]
        if trend_ok and fr > 0 and sr > 0 and (ma_filter <= 0 or c > ma.iloc[i]):
            signals[i] = 1
        elif trend_ok and fr < 0 and sr < 0 and (ma_filter <= 0 or c < ma.iloc[i]):
            signals[i] = -1
        if signals[i] != 0 and not np.isnan(atr14.iloc[i]):
            stops[i] = atr14.iloc[i] * stop_mult

    out["signal"] = signals
    out["stop_dist"] = stops
    out["fast_roc"] = fast_roc
    out["slow_roc"] = slow_roc
    out["ma"] = ma
    out["atr"] = atr14
    return out


def zscore_reversion_signals(
    df: pd.DataFrame,
    window: int = 24,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    ma_filter: int = 24,
    stop_mult: float = 1.5,
) -> pd.DataFrame:
    """Volatility-normalized mean reversion with a trend gate."""
    out = df.copy().reset_index(drop=True)
    close = out["close"]
    mean = sma(close, window)
    std = close.rolling(window).std()
    z = (close - mean) / std
    ma = sma(close, ma_filter) if ma_filter > 0 else None
    atr14 = atr(out["high"], out["low"], close, 14)
    n = len(out)

    signals = np.zeros(n, dtype=int)
    stops = np.zeros(n)
    pos = 0
    for i in range(n):
        zv = z.iloc[i]
        if np.isnan(zv) or (ma_filter > 0 and np.isnan(ma.iloc[i])):
            signals[i] = 0
            pos = 0
            continue
        c = close.iloc[i]
        if pos == 0:
            if zv < -entry_z and (ma_filter <= 0 or c > ma.iloc[i]):
                signals[i] = 1
                pos = 1
            elif zv > entry_z and (ma_filter <= 0 or c < ma.iloc[i]):
                signals[i] = -1
                pos = -1
        elif pos > 0:
            if zv > -exit_z:
                signals[i] = 0
                pos = 0
            else:
                signals[i] = 1
        else:
            if zv < exit_z:
                signals[i] = 0
                pos = 0
            else:
                signals[i] = -1
        if signals[i] != 0 and not np.isnan(atr14.iloc[i]):
            stops[i] = atr14.iloc[i] * stop_mult

    out["signal"] = signals
    out["stop_dist"] = stops
    out["zscore"] = z
    out["ma"] = ma
    out["atr"] = atr14
    return out


def combine_ensemble(frames: list[pd.DataFrame], min_votes: int = 2) -> pd.DataFrame:
    """Majority vote over strategy signal frames."""
    base = frames[0][["date", "open", "high", "low", "close"]].copy()
    n = len(base)
    long_votes = np.zeros(n, dtype=int)
    short_votes = np.zeros(n, dtype=int)
    stop_candidates = np.full(n, np.nan)

    for frame in frames:
        sig = frame["signal"].to_numpy()
        long_votes += (sig > 0).astype(int)
        short_votes += (sig < 0).astype(int)
        stops = frame["stop_dist"].to_numpy()
        for i in range(n):
            if sig[i] != 0 and stops[i] > 0 and (np.isnan(stop_candidates[i]) or stops[i] > stop_candidates[i]):
                stop_candidates[i] = stops[i]

    signals = np.where(
        long_votes >= min_votes,
        1,
        np.where(short_votes >= min_votes, -1, 0),
    )
    stops = np.where((signals != 0) & ~np.isnan(stop_candidates), stop_candidates, 0.0)
    base["signal"] = signals
    base["stop_dist"] = np.nan_to_num(stops)
    return base
