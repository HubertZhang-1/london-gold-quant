# -*- coding: utf-8 -*-
"""Backtest the bull-only single-direction gold grid (2024-2026 + full-history safety).

Converts the screenshot EA into a STRICT BULL-ONLY long strategy:
  only longs, bull-regime gate (bull >= bull_thr), ATR stop + TP, no martingale
  ladder, no reverse hedge, no bear participation. Size is either fixed lot or
  risk-budgeted (lose risk% of balance at the stop distance) for a stable max DD.

Usage: py scripts/backtest_bull_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.bull_grid import BullGridConfig, run_bull_grid_backtest  # noqa: E402
from london_gold.macro_factors import forward_fill_macro, macro_direction_score  # noqa: E402

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)

MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
MACRO["date"] = pd.to_datetime(MACRO["date"], utc=True)
MACRO = MACRO.sort_values("date").reset_index(drop=True)
macro_on_bars = forward_fill_macro(macro_direction_score(MACRO)["macro_score"],
                                   D["date"].to_numpy()).to_numpy()

ERA = (D["date"] >= "2024-01-01") & (D["date"] <= "2026-08-28")
FULL = (D["date"] >= "2019-01-01") & (D["date"] <= "2026-08-28")


def run(mask, cfg, macro=True):
    sub = D[mask].reset_index(drop=True)
    m = macro_on_bars[mask.to_numpy()] if macro else None
    return run_bull_grid_backtest(sub, cfg, macro_series=m)


def fmt(st):
    return (f"ret{st['total_return_pct']:+7.1f}% PF{st['profit_factor']:5.2f} "
            f"maxDD{st['max_drawdown_pct']:5.1f}% win{st['winrate']*100:3.0f}% "
            f"tr{st['trades']:3d} net${st['net_pnl']:+,.0f}")


def line(label, c, mask):
    try:
        res = run(mask, c)
        return f"{label:<30} | {fmt(res['stats'])}"
    except Exception as e:
        return f"{label:<30} | ERR {type(e).__name__}: {e}"


print("=== 牛市单向版 · 风险预算定仓 (risk x% of balance per stop) ===")
print("-" * 84)
print("--- 2024-2026 牛市窗口 ---")
for risk in [0.005, 0.01, 0.02, 0.03]:
    for sm, rr in [(2.5, 2.0), (3.0, 2.0), (3.0, 3.0)]:
        c = BullGridConfig(stop_mult=sm, rr=rr, risk_per_trade_pct=risk)
        print(line(f"risk{risk*100:.1f}% stop{sm} rr{rr}", c, ERA))

print("\n--- 完整区间 2019-2026 (不爆仓校验) ---")
for risk in [0.005, 0.01, 0.02, 0.03]:
    for sm, rr in [(2.5, 2.0), (3.0, 2.0)]:
        c = BullGridConfig(stop_mult=sm, rr=rr, risk_per_trade_pct=risk)
        print(line(f"full risk{risk*100:.1f}% stop{sm} rr{rr}", c, FULL))

# ---- yearly breakdown for the best candidate ----
WINDOWS = [("2019", "2019-01-01", "2019-12-31"), ("2020", "2020-01-01", "2020-12-31"),
           ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
           ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
           ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-28")]
print("\n=== 逐年 (risk 2% stop2.5 rr2.0) ===")
c = BullGridConfig(stop_mult=2.5, rr=2.0, risk_per_trade_pct=0.02)
for y, s, e in WINDOWS:
    m = (D["date"] >= s) & (D["date"] <= e)
    print(f"{y:>5} | " + fmt(run(m, c)["stats"]))
