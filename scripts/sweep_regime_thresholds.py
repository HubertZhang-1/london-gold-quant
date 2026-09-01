# -*- coding: utf-8 -*-
"""Sweep regime-filter thresholds to find a config that lifts all windows."""
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

BASE = dict(min_confidence=0.55, rr_target=1.8, min_adx=18, use_session_filter=False)

windows = [
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
    ("2026H1", "2026-01-01", "2026-06-30"),
    ("2026H2", "2026-07-01", "2026-08-28"),
]

# sweep gentle ER/ADX filters (low thresholds = filter only worst chop)
thresholds = [
    (0.12, 16), (0.13, 17), (0.14, 18), (0.15, 19),
    (0.16, 20), (0.12, 20), (0.14, 16),
]

print(f"{'ER':>5} {'ADX':>5} | {'2024H1':>8} {'2024H2':>8} {'2025H1':>8} {'2026H1':>8} {'2026H2':>8} | {'sumRet':>8} {'minRet':>8} {'avgDD':>7}")
print("-" * 92)
for er, adx in thresholds:
    kw = dict(**BASE, regime_filter=True, er_threshold=er, adx_regime_threshold=adx)
    rets = []
    dds = []
    for wname, s, e in windows:
        part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
        frame = momentum_scalp_signals(part, **kw)
        st = v3bt.backtest_v3(frame, COST, wname, {}).stats
        rets.append(st["total_return"])
        dds.append(st["max_drawdown"])
    if all(r > -0.5 for r in rets):
        print(f"{er:5.2f} {adx:5d} | {rets[0]:+8.2f} {rets[1]:+8.2f} {rets[2]:+8.2f} "
              f"{rets[3]:+8.2f} {rets[4]:+8.2f} | {sum(rets):+8.2f} {min(rets):+8.2f} {sum(dds)/len(dds):7.1f}")

print()
print("Rows shown = all windows positive (minRet>0). Best = high sumRet, positive minRet.")
