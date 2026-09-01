# -*- coding: utf-8 -*-
"""Tune three-line weights/thresholds to balance regime-crossing vs upside."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.gold_system import SystemConfig, build_three_line_frame

spec = importlib.util.spec_from_file_location(
    "v3bt", str(Path(__file__).resolve().parents[1] / "scripts" / "london_gold_intraday_v3_backtest.py"))
v3bt = importlib.util.module_from_spec(spec)
sys.modules["v3bt"] = v3bt
spec.loader.exec_module(v3bt)

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
COT = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\gold_cot_disagg.csv", parse_dates=["date"])
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

WINDOWS = [
    ("2018-20", "2018-01-01", "2020-12-31"),
    ("2024-25", "2024-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-08-28"),
]


def run(part, cfg):
    frame = build_three_line_frame(part, COT, MACRO, cfg)
    return v3bt.backtest_v3(frame, COST, "s", {}).stats


print(f"{'macroW':>6} {'cotW':>5} {'thr':>5} | {'18-20 ret':>9} {'DD':>6} | {'24-25 ret':>9} {'DD':>6} | {'26 ret':>7} {'DD':>5} | {'PF':>5}")
print("-" * 88)
for mk in (0.0, 0.3, 0.6, 0.9):
    for ck in (0.6, 1.0):
        for thr in (0.15, 0.25):
            cfg = SystemConfig(macro_weight=mk, cot_weight=ck, micro_threshold=thr)
            res = {}
            for wname, s, e in WINDOWS:
                part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
                st = run(part, cfg)
                res[wname] = st
            print(f"{mk:6.1f} {ck:5.1f} {thr:5.2f} | {res['2018-20']['total_return']:+9.1f} {res['2018-20']['max_drawdown']:6.1f} | "
                  f"{res['2024-25']['total_return']:+9.1f} {res['2024-25']['max_drawdown']:6.1f} | "
                  f"{res['2026']['total_return']:+7.1f} {res['2026']['max_drawdown']:5.1f} | {res['2026']['profit_factor']:5.2f}")
