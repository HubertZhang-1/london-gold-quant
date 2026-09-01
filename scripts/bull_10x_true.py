# -*- coding: utf-8 -*-
"""True 10x leverage bull-only: let leverage_oz dominate (risk high), wide stop.

Key: nominal = capital*10/price. Width of stop (in % of price) determines real
per-trade risk. Set risk_per_trade_pct high enough that it does NOT cap below
the 10x leverage size, so the account truly runs 10x. Wide ATR stop protects.
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

WINDOWS = [("2019", "2019-01-01", "2019-12-31"), ("2020", "2020-01-01", "2020-12-31"),
           ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
           ("2026", "2026-01-01", "2026-08-28")]


def run(part, thr, stop_mult, bull_thr, risk, mc):
    fac = build_factors(part)
    micro = aggregate_score(fac, MICRO_W)
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=48,
                       er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
    sig = np.where((reg > 0.5) & (micro > thr), 1, np.where((reg > 0.5) & (micro < -thr), -1, 0))
    sig = np.where(part["bull"].to_numpy() > bull_thr, sig, 0)
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy(); anim = ~np.isnan(a)
    f = pd.DataFrame({"date": part["date"], "open": part["open"], "high": part["high"],
                      "low": part["low"], "close": part["close"], "signal": sig,
                      "stop_dist": np.where(anim, a * stop_mult, 0.0),
                      "tp_dist": np.where(anim, a * stop_mult * 2.0, 0.0)})
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, leverage=10.0,
                      risk_per_trade_pct=risk, margin_call_pct=mc)
    return run_leverage_backtest(f, cost, "x")["stats"]


print("=== TRUE 10x leverage bull-only (wide stop, high risk cap) ===")
print(f"{'thr':>4} {'stp':>4} {'risk':>5} {'mc':>4} | {'2019':>6} {'2020':>6} {'2024':>6} {'2025':>6} {'2026':>6}")
print("-" * 68)
for thr in (0.5, 0.7):
    for stop in (3.0, 5.0):
        for risk in (0.20, 0.50):
            for mc in (0.30,):
                row = {}
                for wname, s, e in WINDOWS:
                    part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
                    st = run(part, thr, stop, 0.55, risk, mc)
                    row[wname] = (st["total_return"], st["trade_count"])
                def f(w):
                    return f"{row[w][0]:+6.1f}" if w in row else "  n/a"
                print(f"{thr:4.2f} {stop:4.1f} {risk:5.2f} {mc:4.2f} | {f('2019')} {f('2020')} {f('2024')} {f('2025')} {f('2026')}")
