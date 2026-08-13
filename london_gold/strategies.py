# -*- coding: utf-8 -*-
"""Daily London gold strategies.

Every generator returns a copy of the input frame with at least three extra
columns:

- ``signal``: target position at bar close (-1 / 0 / +1)
- ``stop_dist``: ATR-based stop distance for the signal bar (0 disables)
- ``atr``: 14-bar ATR
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr, donchian, ema, rsi, sma


def _base_frame(df: pd.DataFrame, stop_mult: float) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out["high"], out["low"], out["close"], 14)
    out["stop_dist"] = out["atr"] * stop_mult
    out["signal"] = 0
    return out


def donchian_breakout_signals(
    df: pd.DataFrame,
    entry_n: int = 40,
    exit_n: int = 20,
    ma_filter: int = 100,
    stop_mult: float = 3.0,
) -> pd.DataFrame:
    """Donchian channel breakout with optional trend filter."""
    out = _base_frame(df, stop_mult)
    upper, lower = donchian(out["high"], out["low"], entry_n)
    exit_upper, exit_lower = donchian(out["high"], out["low"], exit_n)
    ma = sma(out["close"], ma_filter) if ma_filter > 0 else pd.Series(np.nan, index=out.index)

    close = out["close"].to_numpy()
    up = upper.to_numpy()
    lo = lower.to_numpy()
    xu = exit_upper.to_numpy()
    xl = exit_lower.to_numpy()
    trend = ma.to_numpy()

    signals = np.zeros(len(out), dtype=int)
    pos = 0
    for i in range(len(out)):
        c = close[i]
        if pos == 0:
            filter_ok = ma_filter <= 0 or not np.isnan(trend[i])
            if not np.isnan(up[i]) and filter_ok and (ma_filter <= 0 or c > trend[i]) and c > up[i]:
                signals[i] = 1
                pos = 1
            elif not np.isnan(lo[i]) and filter_ok and (ma_filter <= 0 or c < trend[i]) and c < lo[i]:
                signals[i] = -1
                pos = -1
        elif pos > 0:
            if not np.isnan(xl[i]) and c < xl[i]:
                signals[i] = 0
                pos = 0
            else:
                signals[i] = 1
        else:
            if not np.isnan(xu[i]) and c > xu[i]:
                signals[i] = 0
                pos = 0
            else:
                signals[i] = -1

    out["signal"] = signals
    out["upper"] = upper
    out["lower"] = lower
    out["exit_upper"] = exit_upper
    out["exit_lower"] = exit_lower
    out["ma"] = ma
    return out


def ema_cross_signals(
    df: pd.DataFrame,
    fast_n: int = 10,
    slow_n: int = 40,
    stop_mult: float = 3.0,
) -> pd.DataFrame:
    """EMA cross trend follower, always in market after the first valid cross."""
    out = _base_frame(df, stop_mult)
    fast = ema(out["close"], fast_n)
    slow = ema(out["close"], slow_n)
    raw = np.where(fast > slow, 1, -1)
    raw[np.isnan(fast) | np.isnan(slow)] = 0
    # ewm is defined from the first bar; keep the slow-window warmup flat.
    raw[: slow_n - 1] = 0
    out["signal"] = raw.astype(int)
    out["fast"] = fast
    out["slow"] = slow
    return out


def rsi_reversal_signals(
    df: pd.DataFrame,
    rsi_n: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    ma_filter: int = 50,
    rsi_exit_long: float = 55.0,
    rsi_exit_short: float = 45.0,
    stop_mult: float = 2.5,
) -> pd.DataFrame:
    """RSI pullback reversal: fade extremes only in the trend direction."""
    out = _base_frame(df, stop_mult)
    rsi_series = rsi(out["close"], rsi_n)
    ma = sma(out["close"], ma_filter) if ma_filter > 0 else pd.Series(np.nan, index=out.index)

    r = rsi_series.to_numpy()
    trend = ma.to_numpy()
    close = out["close"].to_numpy()

    signals = np.zeros(len(out), dtype=int)
    pos = 0
    for i in range(len(out)):
        if np.isnan(r[i]) or (ma_filter > 0 and np.isnan(trend[i])):
            signals[i] = 0
            continue
        if pos == 0:
            if r[i] < oversold and (ma_filter <= 0 or close[i] > trend[i]):
                signals[i] = 1
                pos = 1
            elif r[i] > overbought and (ma_filter <= 0 or close[i] < trend[i]):
                signals[i] = -1
                pos = -1
        elif pos > 0:
            if r[i] > rsi_exit_long or (ma_filter > 0 and close[i] < trend[i]):
                signals[i] = 0
                pos = 0
            else:
                signals[i] = 1
        else:
            if r[i] < rsi_exit_short or (ma_filter > 0 and close[i] > trend[i]):
                signals[i] = 0
                pos = 0
            else:
                signals[i] = -1

    out["signal"] = signals
    out["rsi"] = rsi_series
    out["ma"] = ma
    return out
