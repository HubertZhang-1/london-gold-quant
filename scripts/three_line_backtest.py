# -*- coding: utf-8 -*-
"""Three-line synthetic system backtest: micro + COT + macro vs micro-only.

Runs on 1h XAUUSD continuous data. Compares:
  - micro-only (multi-factor + regime)
  - three-line (micro + COT timing + macro direction)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.cot_factors import compute_cot_factors, cot_timing_score
from london_gold.gold_system import MICRO_WEIGHTS, SystemConfig, build_three_line_frame
from london_gold.indicators import trend_regime

spec = importlib.util.spec_from_file_location(
    "v3bt", str(Path(__file__).resolve().parents[1] / "scripts" / "london_gold_intraday_v3_backtest.py"))
v3bt = importlib.util.module_from_spec(spec)
sys.modules["v3bt"] = v3bt
spec.loader.exec_module(v3bt)

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
COT = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\gold_cot_disagg.csv", parse_dates=["date"]) \
    if Path(r"C:\Users\张策\Documents\EA量化项目\data\gold_cot_disagg.csv").exists() else None
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

WINDOWS = [
    ("2018-20", "2018-01-01", "2020-12-31"),
    ("2021-23", "2021-01-01", "2023-12-31"),
    ("2024-25", "2024-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-08-28"),
]


def run_window(df, macro, cot, s, e, with_cot, with_macro, cfg):
    part = df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)
    frame = build_three_line_frame(part, cot if with_cot else None,
                                   macro if with_macro else None, cfg)
    return v3bt.backtest_v3(frame, COST, "sys", {}).stats


print("=== THREE-LINE SYSTEM vs MICRO-ONLY ===")
for wname, s, e in WINDOWS:
    cfg_micro = SystemConfig()
    cfg_full = SystemConfig()
    # micro-only: turn off COT and macro
    cfg_micro.cot_weight = 0.0
    cfg_micro.macro_weight = 0.0
    micro = run_window(DF, MACRO, COT, s, e, with_cot=False, with_macro=False, cfg=cfg_micro)
    full = run_window(DF, MACRO, COT, s, e, with_cot=True, with_macro=True, cfg=cfg_full)
    print(f"[{wname}] micro-only: ret={micro['total_return']:+.1f}% PF={micro['profit_factor']:.2f} DD={micro['max_drawdown']:.1f}%  |  "
          f"three-line: ret={full['total_return']:+.1f}% PF={full['profit_factor']:.2f} DD={full['max_drawdown']:.1f}%")
