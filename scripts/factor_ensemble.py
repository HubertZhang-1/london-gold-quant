# -*- coding: utf-8 -*-
"""Curated factor ensembles vs equal-weight all-factors, and regime filter."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import FACTOR_BUILDERS, aggregate_score, build_factors
from london_gold.indicators import atr as iatr

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
mask26 = DF["date"] >= "2026-01-01"


def make_signal(score, threshold=0.25, stop_mult=1.0, rr=1.8):
    n = len(score)
    sig = np.where(score > threshold, 1, np.where(score < -threshold, -1, 0))
    a = atr14.to_numpy(float)
    anim = ~np.isnan(a)
    stops = np.where(anim & (sig != 0), a * stop_mult, 0.0)
    takes = np.where(anim & (sig != 0), a * stop_mult * rr, 0.0)
    return pd.DataFrame({
        "date": F["date"], "open": F["open"], "high": F["high"], "low": F["low"], "close": F["close"],
        "signal": sig, "stop_dist": stops, "tp_dist": takes,
    })


# Curated weights: strong trend/momentum factors positive; weak reversal factors dropped or negative.
curated_w = {
    "macd": 1.0, "aroon": 1.0, "trend_adx": 1.0, "ema_spread": 1.0,
    "bulls_bears": 0.8, "momentum": 0.8, "bb_position": 0.5,
}
# Also a version that flips the reversal factors (use them as fade, i.e. negative weight)
flip_w = dict(curated_w)
flip_w["rsi"] = 0.3
flip_w["williams_r"] = 0.3
flip_w["cci"] = 0.2

variants = {
    "all_equal": {c: 1.0 for c in FACTOR_BUILDERS},
    "curated": curated_w,
    "curated_fliprev": flip_w,
}

print(f"{'variant':>18} {'window':>7} {'trades':>6} {'win%':>5} {'ret%':>8} {'PF':>5} {'maxDD%':>6}")
print("-" * 68)
for vname, w in variants.items():
    score = aggregate_score(F, w)
    frame = make_signal(score)
    for wname, part, m in (("2024-25", DF[~mask26], ~mask26), ("2026", DF[mask26], mask26)):
        fr = frame[m].reset_index(drop=True)
        s = v3bt.backtest_v3(fr, COST, vname, {}).stats
        print(f"{vname:>18} {wname:>7} {s['trade_count']:6d} {s['win_rate']:5.1f} "
              f"{s['total_return']:+8.2f} {s['profit_factor']:5.2f} {s['max_drawdown']:6.1f}")
    print()

# full-period with the best (curated)
print("=== FULL PERIOD (aggregate_score with curated weights) ===")
score = aggregate_score(F, curated_w)
frame = make_signal(score)
res = v3bt.backtest_v3(frame, COST, "curated", {})
s = res.stats
print(f"trades={s['trade_count']} win%={s['win_rate']:.1f} ret={s['total_return']:+.2f}% "
      f"PF={s['profit_factor']:.2f} maxDD={s['max_drawdown']:.1f}% final=${s['final_equity']:,.0f}")
