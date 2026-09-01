# -*- coding: utf-8 -*-
"""Find the highest-returning strategy variant within 2025-01..2026-08."""
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
n = len(seg)

# factor weight variants (trend/momentum core)
weight_variants = {
    "curated": {"macd": 1.0, "aroon": 1.0, "trend_adx": 1.0, "ema_spread": 1.0,
                "bulls_bears": 0.8, "momentum": 0.8, "bb_position": 0.5},
    "curated_strong": {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
                       "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4},
    "momentum_heavy": {"macd": 1.2, "aroon": 1.5, "momentum": 1.2, "trend_adx": 1.0,
                       "ema_spread": 0.8},
}


def run_factor(weight, threshold, use_regime, reg_er, reg_adx):
    score = aggregate_score(F, weight)
    sig = np.where(score > threshold, 1, np.where(score < -threshold, -1, 0))
    if use_regime:
        reg = trend_regime(seg["close"], seg["high"], seg["low"], er_window=48,
                           er_threshold=reg_er, adx_window=14, adx_threshold=reg_adx).to_numpy()
        sig = np.where((reg > 0.5) & (sig != 0), sig, 0)
    a = atr14.to_numpy(float)
    anim = ~np.isnan(a)
    frame = pd.DataFrame({
        "date": seg["date"], "open": seg["open"], "high": seg["high"],
        "low": seg["low"], "close": seg["close"],
        "signal": sig, "stop_dist": np.where(anim & (sig != 0), a, 0.0),
        "tp_dist": np.where(anim & (sig != 0), a * 1.8, 0.0)})
    return v3bt.backtest_v3(frame, COST, "x", {}).stats


print("=== FACTOR ENSEMBLE sweep on 2025-01..2026-08 ===")
print(f"{'weights':>16} {'thr':>4} {'reg':>4} {'er':>4} {'adx':>4} | {'trades':>6} {'win%':>5} {'ret%':>9} {'PF':>5} {'maxDD%':>6}")
print("-" * 82)
best = {"ret": -1e9}
for wname, w in weight_variants.items():
    for thr in (0.20, 0.25, 0.30):
        for use_reg in (False, True):
            reg_er = 0.12
            reg_adx = 20
            s = run_factor(w, thr, use_reg, reg_er, reg_adx)
            print(f"{wname:>16} {thr:4.2f} {str(use_reg):>4} {reg_er:4.2f} {reg_adx:4d} | "
                  f"{s['trade_count']:6d} {s['win_rate']:5.1f} {s['total_return']:+9.2f} "
                  f"{s['profit_factor']:5.2f} {s['max_drawdown']:6.1f}")
            if s["total_return"] > best["ret"]:
                best = {"w": wname, "thr": thr, "reg": use_reg, "er": reg_er, "adx": reg_adx,
                        "ret": s["total_return"], "pf": s["profit_factor"],
                        "dd": s["max_drawdown"], "trades": s["trade_count"], "win": s["win_rate"]}
print()
print("=== BEST ===")
print(best)
