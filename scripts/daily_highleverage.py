# -*- coding: utf-8 -*-
"""Daily-bar high-leverage bull strategy: test 3x/5x/10x on Gold daily.

Daily bars have ~1-2.5% average range; at 10x that's 10-25% equity swing per
day if fully sized. We size to risk (risk_per_trade_pct) so a stop (in % of
price) determines real per-trade risk, and use margin_call halt.

Indicator params are daily-appropriate (e.g. ATR window 14 days, trend EMA on
days, bull score on daily bars).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import atr, ema, trend_regime
from london_gold.leverage_backtest import run_leverage_backtest

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)
close = D["close"]
# daily-appropriate long trend (EMA ~200 days, slope over ~5 days)
ema200 = ema(close, 200)
ema_slope = ema200.diff(20)
up_trend = (close > ema(close, 50)).astype(float)
trend = trend_regime(close, D["high"], D["low"], er_window=20, er_threshold=0.12,
                     adx_window=14, adx_threshold=20.0)
D["bull"] = np.clip((ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3, 0, 1)
MICRO_W = {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
           "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4}

WINDOWS = [("2019", "2019-01-01", "2019-12-31"), ("2020", "2020-01-01", "2020-12-31"),
           ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
           ("2026", "2026-01-01", "2026-08-28")]


def run(part, lev, thr=0.5, stop_mult=2.0, rr=2.0, mc=0.30, bull_thr=0.55, risk=0.20):
    fac = build_factors(part)
    micro = aggregate_score(fac, MICRO_W)
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=20,
                       er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
    sig = np.where((reg > 0.5) & (micro > thr), 1, np.where((reg > 0.5) & (micro < -thr), -1, 0))
    sig = np.where(part["bull"].to_numpy() > bull_thr, sig, 0)
    # edge gate on daily
    ms = pd.Series(micro.to_numpy(), index=range(len(part)))
    edge = ms.rolling(20, min_periods=10).mean().fillna(0.0).to_numpy()
    sig = np.where(edge > 0.02, sig, 0)
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy(); anim = ~np.isnan(a)
    f = pd.DataFrame({"date": part["date"], "open": part["open"], "high": part["high"],
                      "low": part["low"], "close": part["close"], "signal": sig,
                      "stop_dist": np.where(anim, a * stop_mult, 0.0),
                      "tp_dist": np.where(anim, a * stop_mult * rr, 0.0)})
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, leverage=lev, risk_per_trade_pct=risk,
                      margin_call_pct=mc)
    return run_leverage_backtest(f, cost, f"lev{lev}")["stats"]


print("=== DAILY-BAR HIGH LEVERAGE BULL (2020/2024/2025/2026) ===")
print(f"{'lev':>4} {'stop':>4} {'risk':>5} | {'2020':>6} {'2024':>6} {'2025':>6} {'2026':>6}")
print("-" * 58)
for lev in (3.0, 5.0, 10.0):
    for stop_mult in (1.5, 2.5):
        for risk in (0.20, 0.50):
            row = {}
            for wname, s, e in WINDOWS:
                part = D[(D["date"] >= s) & (D["date"] <= e)].reset_index(drop=True)
                st = run(part, lev, stop_mult=stop_mult, risk=risk)
                row[wname] = st["total_return"]
            def f(w):
                return f"{row[w]:+6.1f}" if w in row else "  n/a"
            print(f"{lev:4.1f} {stop_mult:4.1f} {risk:5.2f} | {f('2020')} {f('2024')} {f('2025')} {f('2026')}")
