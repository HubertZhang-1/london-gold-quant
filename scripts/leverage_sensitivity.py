# -*- coding: utf-8 -*-
"""Leverage sensitivity: how return and drawdown scale with leverage in bull era.
Shows that high leverage (10x) on gold mostly blows up; 2-4x is the practical cap.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import atr, ema, trend_regime
from london_gold.leverage_backtest import run_leverage_backtest

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
close = DF["close"]
ema200 = ema(close, 4800)
ema_slope = ema200.diff(120)
up_trend = (close > ema(close, 1200)).astype(float)
trend = trend_regime(close, DF["high"], DF["low"], er_window=48, er_threshold=0.12,
                     adx_window=14, adx_threshold=20.0)
DF["bull"] = np.clip((ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3, 0, 1)
MICRO_W = {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
           "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4}
BULL_ERA = DF[(DF["date"] >= "2024-01-01") & (DF["date"] <= "2026-08-28")].reset_index(drop=True)


def run(part, lev, thr, stop_mult, rr, mc):
    fac = build_factors(part)
    micro = aggregate_score(fac, MICRO_W)
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=48,
                       er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
    sig = np.where((reg > 0.5) & (micro > thr), 1, np.where((reg > 0.5) & (micro < -thr), -1, 0))
    sig = np.where(part["bull"].to_numpy() > 0.55, sig, 0)
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy(); anim = ~np.isnan(a)
    f = pd.DataFrame({"date": part["date"], "open": part["open"], "high": part["high"],
                      "low": part["low"], "close": part["close"], "signal": sig,
                      "stop_dist": np.where(anim, a * stop_mult, 0.0),
                      "tp_dist": np.where(anim, a * stop_mult * rr, 0.0)})
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, leverage=lev, risk_per_trade_pct=0.20,
                      margin_call_pct=mc)
    return run_leverage_backtest(f, cost, f"lev{lev}")["stats"]


print("=== LEVERAGE SENSITIVITY in BULL ERA 2024-2026 (bull gate, long+short) ===")
print(f"{'lev':>5} {'stop':>5} | {'trades':>6} {'ret%':>9} {'PF':>5} {'maxDD%':>6} {'final$':>9}")
print("-" * 60)
for lev in (1.0, 2.0, 3.0, 5.0, 10.0):
    for stop in (2.5,):
        st = run(BULL_ERA, lev, 0.5, stop, 2.0, 0.30)
        print(f"{lev:5.1f} {stop:5.1f} | {st['trade_count']:6d} {st['total_return']:+9.1f} "
              f"{st['profit_factor']:5.2f} {st['max_drawdown']:6.1f} {st['final_equity']:9.0f}")
