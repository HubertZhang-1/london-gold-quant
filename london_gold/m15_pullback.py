# -*- coding: utf-8 -*-
"""15-minute momentum breakout with pullback confirmation."""
from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .indicators import atr, ema

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class M15StrategyConfig:
    ema_bars: int = 48
    fast_bars: int = 4
    slow_bars: int = 16
    breakout_bars: int = 8
    pullback_bars: int = 3
    atr_bars: int = 14
    min_body_ratio: float = 0.40
    max_breakout_atr: float = 2.50
    pullback_tolerance_atr: float = 0.35
    min_stop_atr: float = 0.80
    max_stop_atr: float = 2.00


def _in_overlap(timestamp: pd.Timestamp) -> bool:
    utc_time = timestamp.to_pydatetime()
    london_time = utc_time.astimezone(LONDON)
    new_york_time = utc_time.astimezone(NEW_YORK)
    return 8 <= london_time.hour < 17 and 8 <= new_york_time.hour < 17


def _trend_direction(
    close: pd.Series,
    trend: pd.Series,
    fast_roc: pd.Series,
    slow_roc: pd.Series,
    slope_bars: int,
    index: int,
) -> int:
    if index < slope_bars or any(
        np.isnan(value)
        for value in (trend.iloc[index], fast_roc.iloc[index], slow_roc.iloc[index])
    ):
        return 0
    if (
        close.iloc[index] > trend.iloc[index]
        and trend.iloc[index] > trend.iloc[index - slope_bars]
        and fast_roc.iloc[index] > 0
        and slow_roc.iloc[index] > 0
    ):
        return 1
    if (
        close.iloc[index] < trend.iloc[index]
        and trend.iloc[index] < trend.iloc[index - slope_bars]
        and fast_roc.iloc[index] < 0
        and slow_roc.iloc[index] < 0
    ):
        return -1
    return 0


def momentum_pullback_signals(
    df: pd.DataFrame,
    config: M15StrategyConfig | None = None,
) -> pd.DataFrame:
    """Return close-confirmed signals for execution on the next bar open."""
    config = config or M15StrategyConfig()
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    out = df.copy().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    close = pd.to_numeric(out["close"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")

    trend = ema(close, config.ema_bars)
    fast_roc = close.pct_change(config.fast_bars)
    slow_roc = close.pct_change(config.slow_bars)
    atr14 = atr(high, low, close, config.atr_bars)
    breakout_high = high.rolling(config.breakout_bars).max().shift(1)
    breakout_low = low.rolling(config.breakout_bars).min().shift(1)
    session_open = out["date"].map(_in_overlap).astype(bool)

    count = len(out)
    signals = np.zeros(count, dtype=int)
    stop_distances = np.zeros(count, dtype=float)
    states = np.full(count, "idle", dtype=object)
    setup: dict | None = None

    for index in range(count):
        direction = _trend_direction(close, trend, fast_roc, slow_roc, config.fast_bars, index)
        current_atr = float(atr14.iloc[index]) if not np.isnan(atr14.iloc[index]) else 0.0

        if setup is not None:
            setup["age"] += 1
            states[index] = "armed_long" if setup["direction"] > 0 else "armed_short"
            slow_value = slow_roc.iloc[index]
            slow_trend_valid = not np.isnan(slow_value) and (
                (setup["direction"] > 0 and slow_value > 0)
                or (setup["direction"] < 0 and slow_value < 0)
            )
            if (
                not session_open.iloc[index]
                or not slow_trend_valid
                or setup["age"] > config.pullback_bars
                or current_atr <= 0
            ):
                setup = None
                states[index] = "idle"
                continue

            tolerance = current_atr * config.pullback_tolerance_atr
            if direction > 0:
                if close.iloc[index] < setup["level"] - tolerance:
                    setup = None
                    states[index] = "idle"
                    continue
                if low.iloc[index] <= setup["level"] + tolerance:
                    setup["touched"] = True
                    setup["swing"] = min(setup["swing"], float(low.iloc[index]))
                confirmed = setup["touched"] and close.iloc[index] > setup["level"] and close.iloc[index] > open_.iloc[index]
                raw_stop = float(close.iloc[index]) - (setup["swing"] - current_atr * 0.10)
            else:
                if close.iloc[index] > setup["level"] + tolerance:
                    setup = None
                    states[index] = "idle"
                    continue
                if high.iloc[index] >= setup["level"] - tolerance:
                    setup["touched"] = True
                    setup["swing"] = max(setup["swing"], float(high.iloc[index]))
                confirmed = setup["touched"] and close.iloc[index] < setup["level"] and close.iloc[index] < open_.iloc[index]
                raw_stop = (setup["swing"] + current_atr * 0.10) - float(close.iloc[index])

            if confirmed:
                stop_ratio = raw_stop / current_atr
                if config.min_stop_atr <= stop_ratio <= config.max_stop_atr:
                    signals[index] = direction
                    stop_distances[index] = raw_stop
                    states[index] = "confirmed_long" if direction > 0 else "confirmed_short"
                else:
                    states[index] = "invalid_stop"
                setup = None
            continue

        if not session_open.iloc[index] or direction == 0 or current_atr <= 0:
            continue
        bar_range = float(high.iloc[index] - low.iloc[index])
        body_ratio = abs(float(close.iloc[index] - open_.iloc[index])) / bar_range if bar_range > 0 else 0.0
        quality_ok = body_ratio >= config.min_body_ratio and bar_range <= current_atr * config.max_breakout_atr
        if not quality_ok:
            continue

        if direction > 0 and not np.isnan(breakout_high.iloc[index]) and close.iloc[index] > breakout_high.iloc[index]:
            setup = {
                "direction": 1,
                "level": float(breakout_high.iloc[index]),
                "age": 0,
                "touched": False,
                "swing": float(low.iloc[index]),
            }
            states[index] = "armed_long"
        elif direction < 0 and not np.isnan(breakout_low.iloc[index]) and close.iloc[index] < breakout_low.iloc[index]:
            setup = {
                "direction": -1,
                "level": float(breakout_low.iloc[index]),
                "age": 0,
                "touched": False,
                "swing": float(high.iloc[index]),
            }
            states[index] = "armed_short"

    out["signal"] = signals
    out["stop_dist"] = stop_distances
    out["session_open"] = session_open
    out["setup_state"] = states
    out["ema"] = trend
    out["atr"] = atr14
    out["fast_roc"] = fast_roc
    out["slow_roc"] = slow_roc
    out["breakout_high"] = breakout_high
    out["breakout_low"] = breakout_low
    return out
