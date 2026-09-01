# -*- coding: utf-8 -*-
"""Daily high-leverage with STRICT clean-trend selection.
Only trade when the daily trend is very strong & clean (high bull + high
efficiency ratio). Test whether this lets 10x survive more years.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import atr, ema, efficiency_ratio, trend_regime
from london_gold.leverage_backtest import run_leverage_backtest

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)
close = D["close"]
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


def run(part, lev, bull_thr, er_min, stop_mult=1.5, thr=0.6, mc=0.30):
    fac = build_factors(part)
    micro = aggregate_score(fac, MICRO_W)
    er = efficiency_ratio(part["close"], 20)
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=20,
                       er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
    clean = (er.to_numpy() > er_min)  # strong efficiency (clean trend)
    sig = np.where((reg > 0.5) & (micro > thr), 1, np.where((reg > 0.5) & (micro < -thr), -1, 0))
    sig = np.where((part["bull"].to_numpy() > bull_thr) & clean, sig, 0)
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy(); anim = ~np.isnan(a)
    f = pd.DataFrame({"date": part["date"], "open": part["open"], "high": part["high"],
                      "low": part["low"], "close": part["close"], "signal": sig,
                      "stop_dist": np.where(anim, a * stop_mult, 0.0),
                      "tp_dist": np.where(anim, a * stop_mult * 2.0, 0.0)})
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, leverage=lev, risk_per_trade_pct=0.20,
                      margin_call_pct=mc)
    return run_leverage_backtest(f, cost, "x")["stats"]


print("=== DAILY 10x with STRICT clean-trend filter (er_min + bull) ===")
print(f"{'bull':>5} {'erMin':>5} | {'2019':>6} {'2020':>6} {'2024':>6} {'2025':>6} {'2026':>6} | {'#pos':>5}")
print("-" * 62)
for bull_thr in (0.60, 0.70):
    for er_min in (0.18, 0.25, 0.35):
        row = {}
        for wname, s, e in WINDOWS:
            part = D[(D["date"] >= s) & (D["date"] <= e)].reset_index(drop=True)
            st = run(part, 10.0, bull_thr, er_min)
            row[wname] = (st["total_return"], st["trade_count"])
        def f(w):
            return f"{row[w][0]:+6.1f}" if w in row else "  n/a"
        tot = sum(row[w][1] for w in row)
        print(f"{bull_thr:5.2f} {er_min:5.2f} | {f('2019')} {f('2020')} {f('2024')} {f('2025')} {f('2026')} | {tot:5d}")
