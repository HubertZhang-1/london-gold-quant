# -*- coding: utf-8 -*-
"""Combine curated factor ensemble with the regime filter to cut drawdown."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import atr as iatr, trend_regime

spec = importlib.util.spec_from_file_location(
    "v3bt", str(Path(__file__).resolve().parents[1] / "scripts" / "london_gold_intraday_v3_backtest.py"))
v3bt = importlib.util.module_from_spec(spec)
sys.modules["v3bt"] = v3bt
spec.loader.exec_module(v3bt)

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
DF = DF[DF["date"] >= "2024-01-01"].reset_index(drop=True)
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

F = build_factors(DF)
atr14 = iatr(DF["high"], DF["low"], DF["close"], 14)
regime = trend_regime(DF["close"], DF["high"], DF["low"], er_window=48,
                      er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
mask26 = DF["date"] >= "2026-01-01"

curated_w = {
    "macd": 1.0, "aroon": 1.0, "trend_adx": 1.0, "ema_spread": 1.0,
    "bulls_bears": 0.8, "momentum": 0.8, "bb_position": 0.5,
}
score = aggregate_score(F, curated_w)


def make_signal(score, use_regime, threshold=0.25, stop_mult=1.0, rr=1.8):
    n = len(score)
    sig = np.where(score > threshold, 1, np.where(score < -threshold, -1, 0))
    if use_regime:
        sig = np.where((regime > 0.5) & (sig != 0), sig, 0)  # only trade in trend regime
    a = atr14.to_numpy(float)
    anim = ~np.isnan(a)
    stops = np.where(anim & (sig != 0), a * stop_mult, 0.0)
    takes = np.where(anim & (sig != 0), a * stop_mult * rr, 0.0)
    return pd.DataFrame({
        "date": F["date"], "open": F["open"], "high": F["high"], "low": F["low"], "close": F["close"],
        "signal": sig, "stop_dist": stops, "tp_dist": takes,
    })


for use_regime in (False, True):
    frame = make_signal(score, use_regime)
    full = v3bt.backtest_v3(frame, COST, "f", {}).stats
    p1 = v3bt.backtest_v3(frame[~mask26].reset_index(drop=True), COST, "p1", {}).stats
    p2 = v3bt.backtest_v3(frame[mask26].reset_index(drop=True), COST, "p2", {}).stats
    tag = "+regime" if use_regime else "   no-reg"
    print(f"{tag:10s} FULL ret={full['total_return']:+8.2f}% PF={full['profit_factor']:.2f} "
          f"maxDD={full['max_drawdown']:5.1f}% | 2024-25 {p1['total_return']:+8.2f}% "
          f"| 2026 {p2['total_return']:+8.2f}% {p2['max_drawdown']:5.1f}%")
