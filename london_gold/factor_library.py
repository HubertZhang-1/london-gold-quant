# -*- coding: utf-8 -*-
"""Multi-factor library derived from the indicators visible on the MT5 XAUUSD panel.

Each factor is a standardized, direction-consistent score in roughly [-1, +1]:
  +1 = strongly long, -1 = strongly short, 0 = neutral. Every factor is built
  from a single technical indicator (or a small family), matching the MT5
  indicator list (Momentum, RSI, Stochastic, Williams %R, CCI, MACD/OsMA, ADX,
  Bollinger, ATR, Bulls/Bears Power, Aroon, alligator/MA spread, etc.).

We classify factors into families so an ensemble can be built:
  momentum: fast/slow ROC + MACD hist + OsMA direction
  trend:    ADX + EMA slope + Alligator (MA) spread + Aroon
  overbought/oversold: RSI + Stochastic + Williams %R + CCI + BullBears power
  volatility: ATR (normalized) + Bollinger position
  volume:   force index / Chaikin oscillator (requires volume; NA otherwise)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import (
    adx,
    atr,
    bollinger,
    bulls_power,
    bears_power,
    cci,
    sma,
    ema,
    force_index,
    macd,
    momentum,
    roc,
    rsi,
    stochastic,
    williams_r,
    true_range,
)


# ---------------------------------------------------------------------------
# Single-indicator factor builders. Each returns a Series roughly in [-1, 1],
# NaN during warmup. Positive = bullish, negative = bearish.
# ---------------------------------------------------------------------------

def f_momentum_roc(close: pd.Series, fast: int = 6, slow: int = 12) -> pd.Series:
    """Combine fast + slow momentum (rate of change) into a [-1,1] score."""
    f = roc(close, fast)
    s = roc(close, slow)
    # scale by an adaptive std so it's roughly normalized
    scale_f = f.rolling(24 * 24, min_periods=24).std()
    scale_s = s.rolling(24 * 24, min_periods=24).std()
    fc = f / (scale_f + 1e-9)
    sc = s / (scale_s + 1e-9)
    raw = 0.5 * fc + 0.5 * sc
    return np.tanh(raw / 2.0)  # squash to [-1,1]


def f_macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram direction, normalized."""
    line, sig, hist = macd(close, fast, slow, signal)
    scale = hist.rolling(24 * 24, min_periods=24).std()
    return np.tanh((hist / (scale + 1e-9)) / 2.0)


def f_trend_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                adx_window: int = 14, ema_window: int = 50) -> pd.Series:
    """Trend strength signed by price-vs-EMA: only has magnitude in a trend."""
    a = adx(high, low, close, adx_window)
    e = ema(close, ema_window)
    # sign from price relative to longer EMA; magnitude from ADX
    sign = np.sign(close - e)
    mag = np.clip((a - 20.0) / 30.0, 0.0, 1.0)  # 20→0, 50→1
    return (sign * mag).fillna(0.0)


def f_ema_spread(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """EMA fast-slow separation (Alligator/MA spread), direction of trend."""
    f = ema(close, fast)
    s = ema(close, slow)
    spread = (f - s) / close * 1000.0  # in basis points-ish
    scale = spread.rolling(24 * 24, min_periods=24).std()
    return np.tanh((spread / (scale + 1e-9)) / 2.0)


def f_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI reversion-to-mean bias: below 30 -> +, above 70 -> -."""
    r = rsi(close, window)
    # map 30/70 extremes to +1/-1, 50 to 0
    return np.clip((50.0 - r) / 20.0, -1.0, 1.0)


def f_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                 k_window: int = 14) -> pd.Series:
    """Stochastic oscillator reversion bias."""
    k, d = stochastic(high, low, close, k_window=k_window)
    # Oversold k<20 -> +, Overbought k>80 -> -
    return np.clip((50.0 - k) / 30.0, -1.0, 1.0)


def f_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Williams %R reversion bias (near -100 oversold -> +, near 0 overbought -> -)."""
    wr = williams_r(high, low, close, window)
    # wr in [-100,0]; -50 neutral
    return np.clip((-50.0 - wr) / 50.0, -1.0, 1.0)


def f_cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """CCI reversion bias (extreme negative -> +, extreme positive -> -)."""
    c = cci(high, low, close, window)
    return np.clip(-c / 200.0, -1.0, 1.0)


def f_bulls_bears(close: pd.Series, high: pd.Series, low: pd.Series,
                  ema_window: int = 13) -> pd.Series:
    """Bulls vs Bears power balance."""
    bp = bulls_power(close, ema_window)
    brs = bears_power(close, low, ema_window)
    bal = bp + brs  # positive -> buyers; negative -> sellers
    scale = bal.abs().rolling(24 * 24, min_periods=24).mean()
    return np.tanh((bal / (scale + 1e-9)) / 2.0)


def f_atr_vol(high: pd.Series, low: pd.Series, close: pd.Series,
              atr_window: int = 14) -> pd.Series:
    """Volatility factor: current ATR relative to its recent average.
    High vol is regime info, not directional; return magnitude (0..1) signed 0.
    """
    a = atr(high, low, close, atr_window)
    avg = a.rolling(24 * 24, min_periods=24).mean()
    rel = (a / (avg + 1e-9)) - 1.0
    return np.clip(rel, -1.0, 1.0)  # unsigned-ish; used as a neutral gate


def f_bb_position(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger %B position: price at upper -> + (momentum) or mean-revert -."""
    mid, up, lo = bollinger(close, window, num_std)
    width = (up - lo).replace(0, np.nan)
    pb = (close - mid) / width  # %B in [~0,1]
    # reversion bias: over outside bands tends to revert
    return np.clip((0.5 - pb) * 2.0, -1.0, 1.0)


def f_aroon(high: pd.Series, low: pd.Series, close: pd.Series | None = None,
            window: int = 25) -> pd.Series:
    """Aroon trend strength: up-high vs down-low dominance."""
    from .indicators import aroon
    up, down = aroon(high, low, window)
    return np.clip((up - down) / 100.0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Factor registry and ensemble scoring
# ---------------------------------------------------------------------------

FACTOR_BUILDERS = {
    "momentum": f_momentum_roc,
    "macd": f_macd_hist,
    "trend_adx": f_trend_adx,
    "ema_spread": f_ema_spread,
    "rsi": f_rsi,
    "stochastic": f_stochastic,
    "williams_r": f_williams_r,
    "cci": f_cci,
    "bulls_bears": f_bulls_bears,
    "bb_position": f_bb_position,
    "aroon": f_aroon,
}


def build_factors(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Return a DataFrame of factor scores ([-1,1]) plus the raw OHLC.

    Each factor gets its required inputs from the OHLC frame. Factors that need
    volume (force_index / chaikin) are omitted unless the frame has volume.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"] if "volume" in df.columns else None
    out = df[["date", "open", "high", "low", "close"]].copy()
    names = names or list(FACTOR_BUILDERS)

    def loader_for(name: str):
        close_only = {"momentum", "macd", "ema_spread", "rsi", "bb_position"}
        if name in close_only:
            return lambda: FACTOR_BUILDERS[name](close)
        if name == "f_atr_vol":
            return lambda: f_atr_vol(high, low, close)
        # trend_adx / stochastic / williams_r / cci / bulls_bears / aroon
        return lambda: FACTOR_BUILDERS[name](high, low, close)

    for name in names:
        if name not in FACTOR_BUILDERS:
            continue
        if name in ("force_index", "chaikin") and vol is None:
            continue
        out[name] = loader_for(name)()
    return out


def aggregate_score(factors: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Weighted average of named factor columns -> composite [-1,1] score.

    Only factor columns present in BOTH ``factors`` and ``weights`` (or all
    factor columns if weights is None) are used, so a subset weight dict is fine.
    """
    cols = [c for c in FACTOR_BUILDERS if c in factors.columns]
    if not cols:
        raise ValueError("no factor columns")
    weights = weights or {c: 1.0 for c in cols}
    used = [c for c in cols if c in weights]
    if not used:
        raise ValueError("no factor columns overlap weights")
    num = sum(weights[c] * factors[c].fillna(0.0) for c in used)
    den = sum(abs(weights[c]) for c in used) or 1.0
    return np.clip(num / den, -1.0, 1.0)
