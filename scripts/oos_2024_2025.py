# -*- coding: utf-8 -*-
"""Check 2024-2025 1h data completeness and run the v3 best configs on it."""
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

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_5m_1h.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
# 2024-2025 window (the year before our 2026 optimization)
DF = DF[(DF["date"] >= "2024-01-01") & (DF["date"] <= "2025-09-30")].reset_index(drop=True)
print(f"2024-2025 1h: rows={len(DF)}  {DF['date'].min()} -> {DF['date'].max()}")

COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

CONFIGS = {
    "A(0.55,RR1.8,ADX18)": dict(min_confidence=0.55, rr_target=1.8, min_adx=18, use_session_filter=False),
    "B(0.55,RR2.0,ADX25)": dict(min_confidence=0.55, rr_target=2.0, min_adx=25, use_session_filter=False),
    "C(0.60,RR2.0,ADX25)": dict(min_confidence=0.60, rr_target=2.0, min_adx=25, use_session_filter=False),
}

print()
print(f"{'config':>24} {'trades':>6} {'win%':>5} {'ret%':>8} {'PF':>5} {'maxDD%':>6}")
print("-" * 66)
for label, kw in CONFIGS.items():
    frame = momentum_scalp_signals(DF, **kw)
    s = v3bt.backtest_v3(frame, COST, label, {}).stats
    print(f"{label:>24} {s['trade_count']:6d} {s['win_rate']:5.1f} {s['total_return']:+8.2f} "
          f"{s['profit_factor']:5.2f} {s['max_drawdown']:6.1f}")

print()
print("This is the KEY out-of-sample test: 2024-2025 was NOT used to pick the params.")
print("If these configs stay profitable here, the edge is real, not 2026-specific.")
