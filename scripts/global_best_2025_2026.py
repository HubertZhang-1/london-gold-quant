# -*- coding: utf-8 -*-
"""Find global max-return strategy on 2025-01..2026-08 across v3 momentum + factors."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import atr as iatr, trend_regime
from london_gold.intraday_strategies_v3 import momentum_scalp_signals

spec = importlib.util.spec_from_file_location(
    "v3bt", str(Path(__file__).resolve().parents[1] / "scripts" / "london_gold_intraday_v3_backtest.py"))
v3bt = importlib.util.module_from_spec(spec)
sys.modules["v3bt"] = v3bt
spec.loader.exec_module(v3bt)

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
seg = DF[(DF["date"] >= "2025-01-01") & (DF["date"] <= "2026-08-28")].reset_index(drop=True)
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

F = build_factors(seg)
atr14 = iatr(seg["high"], seg["low"], seg["close"], 14)

def run_factor(weight, threshold, stop_mult, rr):
    score = aggregate_score(F, weight)
    sig = np.where(score > threshold, 1, np.where(score < -threshold, -1, 0))
    a = atr14.to_numpy(float); anim = ~np.isnan(a)
    frame = pd.DataFrame({
        "date": seg["date"], "open": seg["open"], "high": seg["high"],
        "low": seg["low"], "close": seg["close"],
        "signal": sig, "stop_dist": np.where(anim & (sig != 0), a * stop_mult, 0.0),
        "tp_dist": np.where(anim & (sig != 0), a * stop_mult * rr, 0.0)})
    return v3bt.backtest_v3(frame, COST, "f", {}).stats

def run_v3(**kw):
    frame = momentum_scalp_signals(seg, **kw)
    return v3bt.backtest_v3(frame, COST, "v3", {}).stats

curated_strong = {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
                  "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4}

print("=== v3 momentum_scalp variants ===")
v3_best = {"ret": -1e9}
for thr in (0.4, 0.5, 0.55):
    for rr in (1.6, 1.8, 2.0, 2.5):
        for stop in (0.8, 1.0, 1.2):
            s = run_v3(min_confidence=thr, rr_target=rr, stop_mult=stop)
            if s["ret" if "ret" in s else "total_return"] > v3_best["ret"]:
                v3_best = {"thr": thr, "rr": rr, "stop": stop,
                           "ret": s["total_return"], "pf": s["profit_factor"],
                           "dd": s["max_drawdown"], "n": s["trade_count"]}
print("v3 best:", v3_best)

print()
print("=== factor ensemble refine (threshold/stop/rr) ===")
fac_best = {"ret": -1e9}
for thr in (0.3, 0.35, 0.4):
    for stop in (0.8, 1.0, 1.2):
        for rr in (1.6, 1.8, 2.2):
            s = run_factor(curated_strong, thr, stop, rr)
            if s["total_return"] > fac_best["ret"]:
                fac_best = {"thr": thr, "stop": stop, "rr": rr,
                            "ret": s["total_return"], "pf": s["profit_factor"],
                            "dd": s["max_drawdown"], "n": s["trade_count"], "win": s["win_rate"]}
print("factor best:", fac_best)
