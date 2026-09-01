# -*- coding: utf-8 -*-
"""Test stricter edge threshold (0.05) vs 0.02 for 3x bull on key years."""
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
c = DF["close"]
ema200 = ema(c, 4800)
slope = ema200.diff(120)
up = (c > ema(c, 1200)).astype(float)
tr = trend_regime(c, DF["high"], DF["low"], er_window=48, er_threshold=0.12, adx_window=14, adx_threshold=20)
DF["bull"] = np.clip((slope > 0).astype(float) * 0.4 + up * 0.3 + tr * 0.3, 0, 1)
W = {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
     "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4}


def run(part, edge_thr):
    fac = build_factors(part)
    micro = aggregate_score(fac, W)
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=48,
                       er_threshold=0.12, adx_window=14, adx_threshold=20).to_numpy()
    sig = np.where((reg > 0.5) & (micro > 0.5), 1, np.where((reg > 0.5) & (micro < -0.5), -1, 0))
    sig = np.where(part["bull"].to_numpy() > 0.55, sig, 0)
    ms = pd.Series(micro.to_numpy(), index=range(len(part)))
    edge = ms.rolling(96, min_periods=48).mean().fillna(0.0).to_numpy()
    sig = np.where(edge > edge_thr, sig, 0)
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy(); anim = ~np.isnan(a)
    f = pd.DataFrame({"date": part["date"], "open": part["open"], "high": part["high"],
                      "low": part["low"], "close": part["close"], "signal": sig,
                      "stop_dist": np.where(anim, a * 2.5, 0.0), "tp_dist": np.where(anim, a * 2.5 * 2.0, 0.0)})
    cost = CostConfig(capital=100000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, leverage=3.0, risk_per_trade_pct=0.20, margin_call_pct=0.30)
    return run_leverage_backtest(f, cost, "x")["stats"]


for w, s, e in [("2020", "2020-01-01", "2020-12-31"), ("2024", "2024-01-01", "2024-12-31"),
                ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-06-01", "2026-08-28")]:
    part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
    r1 = run(part, 0.02)
    r2 = run(part, 0.05)
    print("{}: edge0.02 ret={:+.1f}% dd={:.1f} | edge0.05 ret={:+.1f}% dd={:.1f}".format(
        w, r1["total_return"], r1["max_drawdown"], r2["total_return"], r2["max_drawdown"]))
