# -*- coding: utf-8 -*-
"""Technical indicators used by the London gold strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_down = down.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_up / avg_down
    return 100.0 - 100.0 / (1.0 + rs)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder ATR."""
    return true_range(high, low, close).ewm(alpha=1.0 / window, adjust=False).mean()


def donchian(high: pd.Series, low: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Channel levels excluding the current bar."""
    upper = high.rolling(window).max().shift(1)
    lower = low.rolling(window).min().shift(1)
    return upper, lower
