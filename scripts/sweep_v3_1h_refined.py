# -*- coding: utf-8 -*-
"""Refined v3 1h sweep: tighten around stop=1.0, vary ADX gate and confidence."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.intraday_strategies_v3 import momentum_scalp_signals

spec = importlib.util.spec_from_file_location(
    "v3bt", str(Path(__file__).resolve().parents[1] / "scripts" / "london_gold_intraday_v3_backtest.py"))
v3bt = importlib.util.module_from_spec(spec)
sys.modules["v3bt"] = v3bt
spec.loader.exec_module(v3bt)

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_2026.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

print("=== refined: stop=1.0, vary conf/ADX/RR ===")
print(f"{'conf':>5} {'rr':>4} {'adx':>4} {'sess':>5} | {'trades':>6} {'win%':>5} {'ret%':>8} {'PF':>5} {'maxDD%':>6}")
print("-" * 70)
best = {"ret": -1e9}
for conf in (0.50, 0.55, 0.60):
    for rr in (1.8, 2.0, 2.2, 2.5):
        for adx_min in (18, 22, 25, 30):
            for sess in (True, False):
                frame = momentum_scalp_signals(
                    DF, min_confidence=conf, rr_target=rr, stop_mult=1.0,
                    min_adx=adx_min, use_session_filter=sess,
                )
                s = v3bt.backtest_v3(frame, COST, "M", {}).stats
                if s["trade_count"] < 150:
                    continue
                print(f"{conf:5.2f} {rr:4.1f} {adx_min:4d} {str(sess):>5} | "
                      f"{s['trade_count']:6d} {s['win_rate']:5.1f} {s['total_return']:+8.2f} "
                      f"{s['profit_factor']:5.2f} {s['max_drawdown']:6.1f}")
                if s["total_return"] > best["ret"] and s["max_drawdown"] < 18:
                    best = {"conf": conf, "rr": rr, "adx": adx_min, "sess": sess,
                            "ret": s["total_return"], "pf": s["profit_factor"],
                            "dd": s["max_drawdown"], "trades": s["trade_count"], "win": s["win_rate"]}

print()
print("=== BEST (ret, dd<18%) ===")
print(best)
