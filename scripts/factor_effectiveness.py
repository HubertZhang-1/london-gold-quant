# -*- coding: utf-8 -*-
"""Test the predictive power of each single factor (and the composite) on 1h XAUUSD."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import FACTOR_BUILDERS, aggregate_score, build_factors

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

# Build factor frame
F = build_factors(DF)


def make_signal(factor_series, threshold: float = 0.30, stop_mult: float = 1.0,
                rr: float = 1.8, atr_series=None) -> pd.DataFrame:
    """Turn a factor series into a signal frame with ATR SL/TP."""
    n = len(factor_series)
    sig = np.where(factor_series > threshold, 1,
                   np.where(factor_series < -threshold, -1, 0))
    stops = np.zeros(n)
    takes = np.zeros(n)
    if atr_series is not None:
        a = atr_series.to_numpy(float)
        anim = ~np.isnan(a)
        stops = np.where(anim & (sig != 0), a * stop_mult, 0.0)
        takes = np.where(anim & (sig != 0), a * stop_mult * rr, 0.0)
    return pd.DataFrame({
        "date": F["date"], "open": F["open"], "high": F["high"], "low": F["low"], "close": F["close"],
        "signal": sig, "stop_dist": stops, "tp_dist": takes,
    })


# ATR for sizing stops
from london_gold.indicators import atr as iatr
atr14 = iatr(DF["high"], DF["low"], DF["close"], 14)

# split into 2024-2025 and 2026 to test consistency
mask26 = DF["date"] >= "2026-01-01"

print(f"{'factor':>12} {'2024-25ret':>11} {'2024-25PF':>9} {'2026ret':>9} {'2026PF':>7} {'both+?':>7}")
print("-" * 68)
for name in FACTOR_BUILDERS:
    fs = F[name]
    frame = make_signal(fs, atr_series=atr14)
    frame26 = frame[mask26].reset_index(drop=True)
    frame2425 = frame[~mask26].reset_index(drop=True)
    r1 = v3bt.backtest_v3(frame2425, COST, name, {}).stats
    r2 = v3bt.backtest_v3(frame26, COST, name, {}).stats
    both = "YES" if r1["total_return"] > 0 and r2["total_return"] > 0 else "-"
    print(f"{name:>12} {r1['total_return']:+11.2f} {r1['profit_factor']:9.2f} "
          f"{r2['total_return']:+9.2f} {r2['profit_factor']:7.2f} {both:>7}")

print()
print("composite (equal weight):")
comp = aggregate_score(F)
frame = make_signal(comp, atr_series=atr14)
frame2425 = frame[~mask26].reset_index(drop=True)
frame26 = frame[mask26].reset_index(drop=True)
r1 = v3bt.backtest_v3(frame2425, COST, "composite", {}).stats
r2 = v3bt.backtest_v3(frame26, COST, "composite", {}).stats
print(f"  2024-25: ret={r1['total_return']:+.2f}% PF={r1['profit_factor']:.2f}  "
      f"2026: ret={r2['total_return']:+.2f}% PF={r2['profit_factor']:.2f}")
