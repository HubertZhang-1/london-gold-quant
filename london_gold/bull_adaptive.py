# -*- coding: utf-8 -*-
"""Adaptive-leverage bull strategy with safety circuit breakers (production).

Market state -> (leverage, risk) mapping on DAILY gold bars:
  BEAR        (bull<bull_thr)                            -> 0x,   flat
  EXTREME_VOL (vol_pctl > ext_vol)                       -> 0x,   flat (stand aside)
  HIGH_VOL    (vol_pctl > 0.80)                          -> 1x,   risk 2%
  CLEAN_TREND (er >0.25 & vol<0.65)                      -> 10x,  risk 0.5%
  BULL        (er >0.15)                                 -> 5x,   risk 1%
  CHOP        (else)                                     -> 2x,   risk 2%

Safety:
  - extreme volatility -> stand aside (0 leverage)
  - per-bar risk inverse to leverage (10x->0.5%, 5x->1%, <=2x->2%)
  - margin_call halt on peak drawdown > mc
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import CostConfig
from .factor_library import aggregate_score, build_factors
from .indicators import atr, ema, efficiency_ratio, trend_regime
from .leverage_backtest import run_leverage_backtest

MICRO_W = {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
           "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4}


@dataclass
class AdaptiveConfig:
    capital: float = 100_000.0
    spread: float = 0.35
    slippage: float = 0.10
    commission_per_oz: float = 0.10
    position_oz: float = 10.0
    bull_thr: float = 0.55
    ext_vol_pctl: float = 0.85
    high_vol_pctl: float = 0.80
    er_clean: float = 0.25
    er_bull: float = 0.15
    margin_call_pct: float = 0.20
    micro_thr: float = 0.5
    stop_mult: float = 2.0
    rr: float = 2.0
    # per-trade risk fraction by leverage tier
    risk_10x: float = 0.005
    risk_5x: float = 0.01
    risk_low: float = 0.02


def _risk_for_lev(lev: float, cfg: AdaptiveConfig) -> float:
    if lev >= 10:
        return cfg.risk_10x
    if lev >= 5:
        return cfg.risk_5x
    return cfg.risk_low


def _lev_risk(row, cfg: AdaptiveConfig):
    b = row["bull"]
    er = row["er20"] if not np.isnan(row["er20"]) else 0.0
    vol = row["atr_pctl"] if not np.isnan(row["atr_pctl"]) else 0.5
    if b < cfg.bull_thr:
        return 0.0, 0.0
    if vol > cfg.ext_vol_pctl:
        return 0.0, 0.0
    if vol > cfg.high_vol_pctl:
        return 1.0, _risk_for_lev(1.0, cfg)
    if er > cfg.er_clean and vol < cfg.high_vol_pctl:
        return 10.0, _risk_for_lev(10.0, cfg)
    if er > cfg.er_bull:
        return 5.0, _risk_for_lev(5.0, cfg)
    return 2.0, _risk_for_lev(2.0, cfg)


def prepare_daily(df: pd.DataFrame, cfg: AdaptiveConfig) -> pd.DataFrame:
    """Attach bull/er/atr_pctl/lev/risk columns to a daily OHLC frame."""
    out = df.copy().sort_values("date").reset_index(drop=True)
    close = out["close"]; high = out["high"]; low = out["low"]
    atr14 = atr(high, low, close, 14)
    atr_pct = atr14 / close * 100.0
    atr_pctl = atr_pct.rolling(250, min_periods=120).rank(pct=True)
    er20 = efficiency_ratio(close, 20)
    ema200 = ema(close, 200); ema_slope = ema200.diff(20)
    up_trend = (close > ema(close, 50)).astype(float)
    trend = trend_regime(close, high, low, er_window=20, er_threshold=0.12,
                         adx_window=14, adx_threshold=20).to_numpy()
    bull = np.clip((ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3, 0, 1)
    out["bull"] = bull
    out["er20"] = er20.to_numpy()
    out["atr_pctl"] = atr_pctl.to_numpy()
    rows = [_lev_risk(r, cfg) for _, r in out.iterrows()]
    out["lev"] = [r[0] for r in rows]
    out["risk"] = [r[1] for r in rows]
    return out


def build_signals(df: pd.DataFrame, cfg: AdaptiveConfig) -> pd.DataFrame:
    """Build signal frame (signal/stop_dist/tp_dist) with per-bar lev/risk."""
    fac = build_factors(df)
    micro = aggregate_score(fac, MICRO_W)
    reg = trend_regime(df["close"], df["high"], df["low"], er_window=20,
                       er_threshold=0.12, adx_window=14, adx_threshold=20).to_numpy()
    sig = np.where((reg > 0.5) & (micro > cfg.micro_thr), 1,
                   np.where((reg > 0.5) & (micro < -cfg.micro_thr), -1, 0))
    sig = np.where(df["bull"].to_numpy() > cfg.bull_thr, sig, 0)
    # stand-aside on extreme vol / bear via lev==0
    sig = np.where(df["lev"].to_numpy() > 0, sig, 0)
    a = atr(df["high"], df["low"], df["close"], 14).to_numpy()
    anim = ~np.isnan(a)
    return pd.DataFrame({
        "date": df["date"], "open": df["open"], "high": df["high"],
        "low": df["low"], "close": df["close"], "signal": sig,
        "stop_dist": np.where(anim, a * cfg.stop_mult, 0.0),
        "tp_dist": np.where(anim, a * cfg.stop_mult * cfg.rr, 0.0),
        "lev": df["lev"].to_numpy(), "risk": df["risk"].to_numpy(),
    })


def run_adaptive(df: pd.DataFrame, cfg: AdaptiveConfig) -> dict:
    """Run the adaptive-leverage circuit-breaker strategy and return stats+artifacts."""
    prepared = prepare_daily(df, cfg)
    frame = build_signals(prepared, cfg)
    cost = CostConfig(capital=cfg.capital, position_oz=cfg.position_oz,
                      spread=cfg.spread, slippage=cfg.slippage,
                      commission_per_oz=cfg.commission_per_oz,
                      leverage=3.0, risk_per_trade_pct=cfg.risk_low,
                      margin_call_pct=cfg.margin_call_pct)
    return run_leverage_backtest(frame, cost, "adaptive",
                                 leverage_series=frame["lev"].to_numpy(),
                                 risk_series=frame["risk"].to_numpy())
