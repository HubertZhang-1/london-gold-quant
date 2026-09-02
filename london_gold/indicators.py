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


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    """Momentum (rate-of-change) in points: close - close[window]."""
    return close - close.shift(window)


def roc(close: pd.Series, window: int = 10) -> pd.Series:
    """Rate of change in percent."""
    return close.pct_change(window) * 100.0


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Williams %R (-100..0). Close to 0 = overbought, near -100 = oversold."""
    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    return -100.0 * (highest - close) / (highest - lowest).replace(0, np.nan)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window).mean()
    mad = (tp - sma_tp).abs().rolling(window).mean()
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False).mean()
    hist = line - signal_line
    return line, signal_line, hist


def osc_ma(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """OsMA histogram (MACD line - signal line)."""
    line, signal_line, hist = macd(close, fast, slow, signal)
    return hist


def bulls_power(close: pd.Series, ema_window: int = 13) -> pd.Series:
    """Bulls Power: close - EMA(close). Positive = buyers in control."""
    return close - ema(close, ema_window)


def bears_power(close: pd.Series, low: pd.Series, ema_window: int = 13) -> pd.Series:
    """Bears Power: low - EMA(close). Negative = sellers in control."""
    return low - ema(close, ema_window)


def force_index(close: pd.Series, volume: pd.Series, ema_window: int = 13) -> pd.Series:
    """Force Index (Elder): (close - close[1]) * volume, smoothed."""
    fo = (close.diff() * volume).fillna(0.0)
    return ema(fo, ema_window)


def aroon(high: pd.Series, low: pd.Series, window: int = 25) -> tuple[pd.Series, pd.Series]:
    """Aroon up/down (0..100)."""
    up = high.rolling(window + 1).apply(lambda x: float(np.argmax(x)) / window * 100.0, raw=True)
    down = low.rolling(window + 1).apply(lambda x: float(np.argmin(x)) / window * 100.0, raw=True)
    return up, down


def chaikin_osc(high: pd.Series, low: pd.Series, close: pd.Series,
                volume: pd.Series, fast: int = 3, slow: int = 10) -> pd.Series:
    """Chaikin Oscillator (of accumulation/distribution line)."""
    # Money flow multiplier
    hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm.fillna(0.0) * volume.fillna(0.0)
    adl = mfv.cumsum()
    return ema(adl, fast) - ema(adl, slow)


def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     window: int = 14, atr_window: int = 14) -> pd.Series:
    """Choppiness Index (0..100). High = range/choppy, low = directional trend.

    CHOP = 100 * log10( sum(TR, window) / (max(high,window)-min(low,window)) ) / log10(window)
    Interpretation: > 61.8 = choppy/range, < 38.2 = trending, else neutral.
    """
    tr = true_range(high, low, close)
    hh = high.rolling(window).max()
    ll = low.rolling(window).min()
    denom = (hh - ll).replace(0, np.nan)
    ratio = tr.rolling(window).sum() / denom
    out = 100.0 * (np.log10(ratio) / np.log10(window))
    return out.clip(0.0, 100.0)


def market_state(close: pd.Series, high: pd.Series, low: pd.Series,
                 er_window: int = 20, er_thr: float = 0.15,
                 adx_window: int = 14, adx_thr: float = 20.0,
                 chop_window: int = 14, chop_hi: float = 61.8, chop_lo: float = 38.2,
                 ema_window: int = 50) -> pd.DataFrame:
    """Classify each bar's market RHYTHM into a state + direction -> a buy/sell signal.

    This is the "行情节奏识别器" (market-rhythm identifier): it distinguishes a
    clean directional trend from a choppy/range market, and returns which way the
    trend points. It composes three independent measures:
      - Choppiness Index  : high -> range (choppy), low -> trending
      - Efficiency ratio  : high -> clean directional, low -> noisy chop
      - ADX               : trend strength
    Final direction comes from the EMA slope, cross-checked by price-vs-EMA.

    Returns a DataFrame with:
      state   : "trend" | "chop" | "neutral"
      dir_    : +1 (up), -1 (down), 0 (flat)
      signal  : the tradeable signal (+1 long-only trend, -1 short, 0 stay flat)
      chop    : raw choppiness index
      er      : efficiency ratio
      adx     : ADX
    """
    chop = choppiness_index(high, low, close, window=chop_window)
    er = efficiency_ratio(close, er_window)
    adx_ = adx(high, low, close, adx_window)
    ema_ = ema(close, ema_window)
    slope = ema_.diff(max(1, ema_window // 5))

    trending = (chop < chop_hi) & (chop > 0) & (er >= er_thr) & (adx_ >= adx_thr)
    choppy = (chop >= chop_hi) | (er < 0.5 * er_thr) | (adx_ < 0.5 * adx_thr)

    dir_ = np.where(slope > 0, 1, np.where(slope < 0, -1, 0))
    # only up-trend counts as long-friendly for a bull-only system
    state = np.where(trending, "trend", np.where(choppy, "chop", "neutral"))
    # signal: long only when in a confirmed UP trend; flat otherwise.
    signal = np.where((trending) & (dir_ > 0), 1,
                      np.where((trending) & (dir_ < 0), -1, 0))
    return pd.DataFrame({
        "state": state, "dir": dir_, "signal": signal,
        "chop": chop, "er": er, "adx": adx_,
    })
