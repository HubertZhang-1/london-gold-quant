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


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder ADX (trend strength 0-100)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1.0 / window, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / window, adjust=False).mean() / atr_.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / window, adjust=False).mean() / atr_.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / window, adjust=False).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_window: int = 14, k_smooth: int = 3, d_window: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D (0-100)."""
    lowest = low.rolling(k_window).min()
    highest = high.rolling(k_window).max()
    raw_k = 100.0 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    k = raw_k.rolling(k_smooth).mean()
    d = k.rolling(d_window).mean()
    return k, d


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (mid, upper, lower)."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def efficiency_ratio(close: pd.Series, window: int = 48) -> pd.Series:
    """Kaufman efficiency ratio: |net move| / sum(|1-bar moves|), 0..1.

    High = clean directional trend (small noise), low = choppy. Useful as a
    trend-vs-noise regime filter for momentum strategies.
    """
    c = close.to_numpy(float)
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(window, n):
        net = abs(c[i] - c[i - window])
        gross = np.abs(np.diff(c[i - window:i + 1])).sum()
        out[i] = net / gross if gross > 0 else 0.0
    return pd.Series(out, index=close.index)


def trend_regime(close: pd.Series, high: pd.Series, low: pd.Series,
                 er_window: int = 48, er_threshold: float = 0.18,
                 adx_window: int = 14, adx_threshold: float = 20.0) -> pd.Series:
    """Return 1 when the market is in a tradeable 'trend' regime (both the
    efficiency ratio and ADX agree), 0 otherwise. Used to gate momentum entries."""
    er = efficiency_ratio(close, er_window)
    adx_ = adx(high, low, close, adx_window)
    mask = (er >= er_threshold) & (adx_ >= adx_threshold)
    return mask.astype(float)
