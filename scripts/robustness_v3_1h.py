# -*- coding: utf-8 -*-
"""Robustness check: split 1h data into two halves, confirm best config is stable."""
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

# candidate configurations (conf, rr, adx, sess)
cands = {
    "A (0.55,1.8,18,no)": dict(min_confidence=0.55, rr_target=1.8, min_adx=18, use_session_filter=False),
    "B (0.55,2.0,25,no)": dict(min_confidence=0.55, rr_target=2.0, min_adx=25, use_session_filter=False),
    "C (0.60,2.0,25,no)": dict(min_confidence=0.60, rr_target=2.0, min_adx=25, use_session_filter=False),
    "D (0.50,2.0,25,no)": dict(min_confidence=0.50, rr_target=2.0, min_adx=25, use_session_filter=False),
}

half1 = DF[DF["date"] < pd.Timestamp("2026-05-01", tz="UTC")].reset_index(drop=True)
half2 = DF[DF["date"] >= pd.Timestamp("2026-05-01", tz="UTC")].reset_index(drop=True)

print(f"{'config':>26} {'window':>8} | {'trades':>6} {'win%':>5} {'ret%':>8} {'PF':>5} {'maxDD%':>6}")
print("-" * 78)
for label, kw in cands.items():
    for wname, part in (("H1<=04", half1), ("H2>04", half2)):
        frame = momentum_scalp_signals(part, **kw)
        s = v3bt.backtest_v3(frame, COST, label, {}).stats
        print(f"{label:>26} {wname:>8} | {s['trade_count']:6d} {s['win_rate']:5.1f} "
              f"{s['total_return']:+8.2f} {s['profit_factor']:5.2f} {s['max_drawdown']:6.1f}")

print()
print("Note: half-windows have no EMA/ATR warmup, so early bars are relative;")
print("compare half1 vs half2 trend (both positive = robust; flip = overfit).")
