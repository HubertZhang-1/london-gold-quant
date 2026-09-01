# -*- coding: utf-8 -*-
"""Rolling out-of-sample: walk config A across 2024-2025 in 6-month windows."""
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

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
DF = DF[DF["date"] >= "2024-01-01"].reset_index(drop=True)
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

# Use config A, the robust one
KW = dict(min_confidence=0.55, rr_target=1.8, min_adx=18, use_session_filter=False)

# 6-month rolling windows (with overlaps to observe stability)
windows = [
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
    ("2025H2", "2025-07-01", "2025-12-31"),
    ("2026H1", "2026-01-01", "2026-06-30"),
    ("2026H2", "2026-07-01", "2026-08-28"),
]

print(f"{'window':>8} {'rows':>6} {'trades':>6} {'win%':>5} {'ret%':>8} {'PF':>5} {'maxDD%':>6}")
print("-" * 60)
for wname, s, e in windows:
    part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
    if len(part) < 100:
        print(f"{wname:>8} {len(part):6d}  (insufficient data)")
        continue
    frame = momentum_scalp_signals(part, **KW)
    res = v3bt.backtest_v3(frame, COST, wname, {})
    s_ = res.stats
    print(f"{wname:>8} {len(part):6d} {s_['trade_count']:6d} {s_['win_rate']:5.1f} "
          f"{s_['total_return']:+8.2f} {s_['profit_factor']:5.2f} {s_['max_drawdown']:6.1f}")

print()
print("Interpretation: positive ret across MOST windows = real, market-regime-robust edge.")
print("A few negative windows is normal for any trend strategy; ALL negative = no edge.")
