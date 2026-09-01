# -*- coding: utf-8 -*-
"""Independent out-of-sample test of the curated multi-factor + regime filter
using 2018-2023 (none of these years informed the factor weights/thresholds)."""
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
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

CURATED = {"macd": 1.0, "aroon": 1.0, "trend_adx": 1.0, "ema_spread": 1.0,
           "bulls_bears": 0.8, "momentum": 0.8, "bb_position": 0.5}


def run_year(part, use_regime, label):
    F = build_factors(part)
    atr14 = iatr(part["high"], part["low"], part["close"], 14)
    score = aggregate_score(F, CURATED)
    n = len(score)
    threshold = 0.25
    sig = np.where(score > threshold, 1, np.where(score < -threshold, -1, 0))
    if use_regime:
        reg = trend_regime(part["close"], part["high"], part["low"], er_window=48,
                           er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
        sig = np.where((reg > 0.5) & (sig != 0), sig, 0)
    a = atr14.to_numpy(float)
    anim = ~np.isnan(a)
    stops = np.where(anim & (sig != 0), a * 1.0, 0.0)
    takes = np.where(anim & (sig != 0), a * 1.0 * 1.8, 0.0)
    frame = pd.DataFrame({
        "date": part["date"], "open": part["open"], "high": part["high"],
        "low": part["low"], "close": part["close"],
        "signal": sig, "stop_dist": stops, "tp_dist": takes,
    })
    s = v3bt.backtest_v3(frame, COST, label, {}).stats
    return s


print("=== INDEPENDENT OOS: curated multi-factor + regime filter, 2018-2023 ===")
print(f"{'year':>6} {'trades':>6} {'win%':>5} {'ret%':>8} {'PF':>5} {'maxDD%':>6}")
print("-" * 56)
rows = []
for y in range(2018, 2024):
    part = DF[(DF["date"] >= f"{y}-01-01") & (DF["date"] <= f"{y}-12-31")].reset_index(drop=True)
    s = run_year(part, use_regime=True, label=str(y))
    rows.append((y, s))
    print(f"{y:>6} {s['trade_count']:6d} {s['win_rate']:5.1f} {s['total_return']:+8.2f} "
          f"{s['profit_factor']:5.2f} {s['max_drawdown']:6.1f}")

print()
print("=== AGGREGATE 2018-2023 (all years joined, single run) ===")
all181 = DF[(DF["date"] >= "2018-01-01") & (DF["date"] <= "2023-12-31")].reset_index(drop=True)
s_all = run_year(all181, use_regime=True, label="2018-23")
print(f"trades={s_all['trade_count']} win%={s_all['win_rate']:.1f} "
      f"ret={s_all['total_return']:+.2f}% PF={s_all['profit_factor']:.2f} "
      f"maxDD={s_all['max_drawdown']:.1f}% final=${s_all['final_equity']:,.0f}")

print()
print("=== COMPARISON on 2018-2023: curated vs all-factor equal weight (regime on) ===")
s_cur = s_all
# all-factor equal weight
F = build_factors(all181)
score_eq = aggregate_score(F)  # all factors equal weight
atr14 = iatr(all181["high"], all181["low"], all181["close"], 14)
sig = np.where(score_eq > 0.25, 1, np.where(score_eq < -0.25, -1, 0))
reg = trend_regime(all181["close"], all181["high"], all181["low"], er_window=48,
                   er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
sig = np.where((reg > 0.5) & (sig != 0), sig, 0)
a = atr14.to_numpy(float)
anim = ~np.isnan(a)
frame = pd.DataFrame({
    "date": all181["date"], "open": all181["open"], "high": all181["high"],
    "low": all181["low"], "close": all181["close"],
    "signal": sig, "stop_dist": np.where(anim & (sig != 0), a, 0.0),
    "tp_dist": np.where(anim & (sig != 0), a * 1.8, 0.0)})
s_eq = v3bt.backtest_v3(frame, COST, "eq", {}).stats
print(f"curated: ret={s_cur['total_return']:+.2f}% PF={s_cur['profit_factor']:.2f} maxDD={s_cur['max_drawdown']:.1f}%")
print(f"all-eq : ret={s_eq['total_return']:+.2f}% PF={s_eq['profit_factor']:.2f} maxDD={s_eq['max_drawdown']:.1f}%")

# save equity
eq = pd.DataFrame({"date": all181["date"], "equity": v3bt.backtest_v3(
    pd.DataFrame({"date": all181["date"], "open": all181["open"], "high": all181["high"],
                  "low": all181["low"], "close": all181["close"],
                  "signal": sig, "stop_dist": np.where(anim & (sig != 0), a, 0.0),
                  "tp_dist": np.where(anim & (sig != 0), a * 1.8, 0.0)}), COST, "x", {}).equity})
out = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "xauusd_1h_oos_2018_2023"
pd.DataFrame({"date": all181["date"],
              "equity": v3bt.backtest_v3(pd.DataFrame({
                  "date": all181["date"], "open": all181["open"], "high": all181["high"],
                  "low": all181["low"], "close": all181["close"],
                  "signal": sig, "stop_dist": np.where(anim & (sig != 0), a, 0.0),
                  "tp_dist": np.where(anim & (sig != 0), a * 1.8, 0.0)}), COST, "x", {}).equity
}).to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved {out}_equity.csv")
