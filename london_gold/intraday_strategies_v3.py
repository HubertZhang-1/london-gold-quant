# -*- coding: utf-8 -*-
"""Third-generation intraday gold strategies.

Borrows the strongest ideas from open-source XAUUSD systems (ns-vikas scalper,
Shivkeerth breakout, session-dynamics research) and the local toolkit:

  - multi-indicator CONFIDENCE SCORING (RSI/EMA/Stoch/ADX/BB), threshold-gated
  - ADX trend-strength gate (weak trends are filtered; strong trend blocks mean-rev)
  - SESSION filter (London/NY/overlap preferred, Asia quiet)
  - fixed ATR-based risk:reward (SL = 1*ATR, TP = 2*ATR) with dollar-risk sizing hook
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import adx, atr, bollinger, ema, rsi, sma, stochastic, trend_regime

# (start_hour, end_hour, weight) in UTC. Overlap weighted highest.
SESSIONS_UTC = {
    "asia": (0, 7, 0.3),
    "london": (7, 13, 1.0),
    "ny": (13, 21, 1.0),
    "overlap": (13, 16, 1.2),  # London-NY overlap
    "quiet": (21, 24, 0.2),
}


def _session_weight(hour: int) -> float:
    for name, (start, end, w) in SESSIONS_UTC.items():
        if start <= hour < end:
            return w
    return 0.2


def momentum_scalp_signals(
    df: pd.DataFrame,
    fast_bars: int = 6,
    slow_bars: int = 24,
    ema_bars: int = 21,
    adx_bars: int = 14,
    min_adx: float = 18.0,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    min_confidence: float = 0.55,
    rr_target: float = 1.8,          # TP = rr_target * SL (optimized on 1h: 1.8-2.0)
    stop_mult: float = 1.0,          # SL = stop_mult * ATR (tight stop wins on 1h)
    use_session_filter: bool = False,
    min_session_weight: float = 0.5,
    regime_filter: bool = True,      # gate entries to trend regimes (default ON)
    er_window: int = 48,
    er_threshold: float = 0.12,
    adx_regime_threshold: float = 20.0,
) -> pd.DataFrame:
    """Dual-momentum entry with confidence score + ADX gate + session filter.

    Long conditions (weighted): fast ROC>0, slow ROC>0, price>EMA(21),
      RSI in (oversold, 70), Stoch%K>%D, ADX>min_adx.
    Confidence = weighted agreement; entries only above threshold.

    When ``regime_filter`` is True, entries are suppressed unless the market is
    in a trend regime (efficiency-ratio AND longer-context ADX both above their
    thresholds). This is meant to avoid the choppy/mean-reverting stretches
    (e.g. 2024H2) where the momentum edge disappears.
    """
    out = df.copy().reset_index(drop=True)
    close = out["close"]
    high = out["high"]
    low = out["low"]
    fast_roc = close.pct_change(fast_bars)
    slow_roc = close.pct_change(slow_bars)
    ema21 = ema(close, ema_bars)
    rsi14 = rsi(close, 14)
    stoch_k, stoch_d = stochastic(high, low, close)
    adx14 = adx(high, low, close, adx_bars)
    atr14 = atr(high, low, close, 14)
    hours = out["date"].dt.hour.to_numpy()
    regime = trend_regime(close, high, low, er_window, er_threshold,
                          adx_window=adx_bars, adx_threshold=adx_regime_threshold).to_numpy() if regime_filter else None

    n = len(out)
    signals = np.zeros(n, dtype=int)
    stops = np.zeros(n)
    takes = np.zeros(n)
    conf = np.zeros(n)

    for i in range(n):
        fr, sr = fast_roc.iloc[i], slow_roc.iloc[i]
        if np.isnan(fr) or np.isnan(sr):
            continue
        c = close.iloc[i]
        a = adx14.iloc[i]
        if np.isnan(a) or np.isnan(ema21.iloc[i]):
            continue

        # regime gate: suppress entries outside a clean trend regime
        if regime is not None and (np.isnan(regime[i]) or regime[i] < 0.5):
            signals[i] = 0
            stops[i] = 0.0
            takes[i] = 0.0
            conf[i] = 0.0
            continue

        h = hours[i]
        sess_w = _session_weight(h)

        for direction, name in ((1, "long"), (-1, "short")):
            score = 0.0
            if direction > 0:
                # momentum + trend + mean-reversion-confirmation
                if fr > 0:
                    score += 0.3
                if sr > 0:
                    score += 0.2
                if c > ema21.iloc[i]:
                    score += 0.2
                if rsi_oversold < rsi14.iloc[i] < 60:
                    score += 0.15
                if not np.isnan(stoch_k.iloc[i]) and stoch_k.iloc[i] > stoch_d.iloc[i]:
                    score += 0.15
            else:
                if fr < 0:
                    score += 0.3
                if sr < 0:
                    score += 0.2
                if c < ema21.iloc[i]:
                    score += 0.2
                if 40 < rsi14.iloc[i] < rsi_overbought:
                    score += 0.15
                if not np.isnan(stoch_k.iloc[i]) and stoch_k.iloc[i] < stoch_d.iloc[i]:
                    score += 0.15

            # ADX gate: require trend strength to act on momentum
            if a < min_adx:
                score *= 0.45  # weaken not reject (sideways still slightly allowed)

            # session gate
            if use_session_filter and sess_w < min_session_weight:
                score *= 0.4

            conf[i] = max(conf[i], score)
            if score >= min_confidence:
                # prefer direction with higher score
                if direction > 0 and (signals[i] == 0 or score > conf[i]):
                    signals[i] = 1
                elif direction < 0 and (signals[i] == 0 or score > conf[i]):
                    signals[i] = -1

        if signals[i] != 0 and not np.isnan(atr14.iloc[i]):
            sl = atr14.iloc[i] * stop_mult
            stops[i] = sl
            takes[i] = sl * rr_target

    out["signal"] = signals
    out["stop_dist"] = stops
    out["tp_dist"] = takes
    out["confidence"] = conf
    out["regime"] = np.nan if regime is None else regime
    out["fast_roc"] = fast_roc
    out["slow_roc"] = slow_roc
    out["ema21"] = ema21
    out["rsi14"] = rsi14
    out["adx14"] = adx14
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d
    out["atr"] = atr14
    return out


def mean_reversion_signals(
    df: pd.DataFrame,
    window: int = 24,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    ma_filter: int = 24,
    adx_bars: int = 14,
    max_adx_meanrev: float = 32.0,
    min_session_weight: float = 0.5,
    stop_mult: float = 1.5,
    rr_target: float = 2.0,
    use_session_filter: bool = True,
) -> pd.DataFrame:
    """Mean reversion gated by: no strong trend (ADX < max), trend filter, session."""
    out = df.copy().reset_index(drop=True)
    close = out["close"]
    high = out["high"]
    low = out["low"]
    mean = sma(close, window)
    std = close.rolling(window).std()
    z = (close - mean) / std
    ma = sma(close, ma_filter) if ma_filter > 0 else None
    adx14 = adx(high, low, close, adx_bars)
    atr14 = atr(high, low, close, 14)
    hours = out["date"].dt.hour.to_numpy()
    n = len(out)

    signals = np.zeros(n, dtype=int)
    stops = np.zeros(n)
    takes = np.zeros(n)
    pos = 0

    for i in range(n):
        zv = z.iloc[i]
        a = adx14.iloc[i]
        if np.isnan(zv) or np.isnan(a):
            signals[i] = 0
            pos = 0
            continue
        c = close.iloc[i]
        sess_w = _session_weight(hours[i])

        # mean reversion is only allowed when NOT in a strong trend
        too_trendy = a > max_adx_meanrev
        session_ok = (not use_session_filter) or sess_w >= min_session_weight

        if pos == 0:
            entry_long = zv < -entry_z and (ma_filter <= 0 or c > ma.iloc[i])
            entry_short = zv > entry_z and (ma_filter <= 0 or c < ma.iloc[i])
            if session_ok and not too_trendy and entry_long:
                signals[i] = 1
                pos = 1
            elif session_ok and not too_trendy and entry_short:
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
            sl = atr14.iloc[i] * stop_mult
            stops[i] = sl
            takes[i] = sl * rr_target

    out["signal"] = signals
    out["stop_dist"] = stops
    out["tp_dist"] = takes
    out["zscore"] = z
    out["adx14"] = adx14
    out["atr"] = atr14
    return out
