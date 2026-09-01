# -*- coding: utf-8 -*-
"""Final three-line chosen configs vs micro-only, full 2018-2026 windows."""
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
    ("2018", "2018-01-01", "2018-12-31"),
    ("2019", "2019-01-01", "2019-12-31"),
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-08-28"),
]


def run(part, cfg, use_cot=True, use_macro=True):
    frame = build_three_line_frame(part, COT if use_cot else None,
                                   MACRO if use_macro else None, cfg)
    return v3bt.backtest_v3(frame, COST, "s", {}).stats


# micro-only (no COT, no macro)
cfg_micro = SystemConfig(cot_weight=0.0, macro_weight=0.0)
# chosen three-line: COT-heavy, light macro
cfg_3line = SystemConfig(cot_weight=1.0, macro_weight=0.3, micro_threshold=0.15)

print(f"{'year':>5} | {'micro ret':>9} {'micro DD':>8} | {'3line ret':>9} {'3line DD':>7} {'3line PF':>7}")
print("-" * 66)
accum_m = {"ret": 0.0, "trades": 0, "wins": 0}
for wname, s, e in WINDOWS:
    part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
    m = run(part, cfg_micro, use_cot=False, use_macro=False)
    t = run(part, cfg_3line, use_cot=True, use_macro=True)
    print(f"{wname:>5} | {m['total_return']:+9.1f} {m['max_drawdown']:8.1f} | "
          f"{t['total_return']:+9.1f} {t['max_drawdown']:7.1f} {t['profit_factor']:7.2f}")
