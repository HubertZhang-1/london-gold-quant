# -*- coding: utf-8 -*-
"""Three-line synthetic gold trading system.

Merges three independent, cross-validated signal layers:
  - MICRO  : multi-factor score (technical, project-verified) gated by trend regime
  - MEDIUM : CFTC COT timing score (positioning)
  - MACRO  : real-yield/USD/VIX direction score (wind vane)

Synthesis (multi-period weighted score) decides direction and position size.
Per the Dongzheng multi-period research, combined signals beat single-period.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cot_factors import compute_cot_factors, cot_timing_score
from .factor_library import aggregate_score, build_factors
from .indicators import atr, trend_regime
from .macro_factors import forward_fill_macro, macro_direction_score

# Micro weights (verified on 2024-26, product of momentum/trend factors)
MICRO_WEIGHTS = {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
                 "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4}


@dataclass
class SystemConfig:
    micro_weight: float = 1.0
    cot_weight: float = 1.0
    macro_weight: float = 0.6       # macro is background, lower weight
    micro_threshold: float = 0.25   # entry gate for micro score
    cot_min_votes: int = 2          # entry gate for COT vote
    macro_gate: float = 0.25        # above this macro score, allow longs etc.
    regime_er: float = 0.12
    regime_adx: float = 20.0
    stop_mult: float = 0.8
    rr: float = 2.2
    risk_per_trade_pct: float = 0.01
    cooldown: int = 0               # bars between re-entries


def build_three_line_frame(
    df: pd.DataFrame,
    cot: pd.DataFrame,
    macro: pd.DataFrame,
    config: SystemConfig | None = None,
) -> pd.DataFrame:
    """Return a signal frame with composite direction + risk sizing columns."""
    config = config or SystemConfig()
    n = len(df)

    # ---- MICRO: factor score + regime gate ----
    fac = build_factors(df)
    micro_score = aggregate_score(fac, MICRO_WEIGHTS)
    regime = trend_regime(df["close"], df["high"], df["low"],
                          er_window=48, er_threshold=config.regime_er,
                          adx_window=14, adx_threshold=config.regime_adx).to_numpy()

    # ---- MACRO: direction score forward-filled onto bars ----
    if macro is not None and len(macro):
        macro_score_series = macro_direction_score(macro)["macro_score"]
        macro_on_bars = forward_fill_macro(macro_score_series, df["date"].to_numpy())
    else:
        macro_on_bars = pd.Series(0.0, index=range(len(df)))

    # ---- MEDIUM: COT timing score forward-filled onto bars ----
    cot_score = _cot_score_on_bars(cot, df["date"].to_numpy())

    # ---- SYNTHESIS: weighted composite in [-1,1] ----
    wm, wc, wk = config.micro_weight, config.cot_weight, config.macro_weight
    # convert all to numpy aligned arrays to avoid pandas index broadcasting
    micro_arr = micro_score.to_numpy()
    cot_arr = cot_score.to_numpy()
    macro_arr = macro_on_bars.to_numpy()
    composite = (wm * micro_arr + wc * cot_arr + wk * macro_arr) / (wm + wc + wk)

    # regime: only allow directional trades in a trend regime
    direction = np.where(
        (regime > 0.5) & (composite > config.micro_threshold), 1,
        np.where((regime > 0.5) & (composite < -config.micro_threshold), -1, 0))

    # position sizing: scale by |composite| and regime, capped
    size = np.abs(composite) * (regime > 0.5)
    atr14 = atr(df["high"], df["low"], df["close"], 14)
    stop = np.where(direction != 0, atr14.to_numpy() * config.stop_mult, 0.0)
    tp = np.where(direction != 0, atr14.to_numpy() * config.stop_mult * config.rr, 0.0)

    out = df[["date", "open", "high", "low", "close"]].copy()
    out["signal"] = direction
    out["stop_dist"] = np.nan_to_num(stop)
    out["tp_dist"] = np.nan_to_num(tp)
    out["size"] = np.nan_to_num(size)
    out["micro"] = micro_arr
    out["macro"] = macro_arr
    out["cot"] = cot_arr
    out["composite"] = composite
    out["regime"] = regime
    return out


def _cot_score_on_bars(cot: pd.DataFrame, bar_dates) -> pd.Series:
    """Build COT timing score and forward-fill onto higher-frequency bar dates."""
    if cot is None or len(cot) == 0:
        return pd.Series(0.0, index=range(len(bar_dates)))
    factors = compute_cot_factors(cot)
    votes = cot_timing_score_votes(factors)
    score = cot_timing_score(votes)
    # ensure tz-aware UTC index on the weekly COT score
    cot_idx = pd.DatetimeIndex(pd.to_datetime(cot["date"].values, utc=True))
    s = pd.Series(score.values, index=cot_idx)
    # normalize bar_dates to tz-aware index
    target = pd.DatetimeIndex(bar_dates)
    if target.tz is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    return s.reindex(target, method="ffill").fillna(0.0)


def cot_timing_score_votes(factors: pd.DataFrame) -> pd.DataFrame:
    """Import the vote frame builder (kept here to avoid circular import)."""
    from .cot_factors import cot_5factor_vote_factors
    return cot_5factor_vote_factors(factors)
