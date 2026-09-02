# -*- coding: utf-8 -*-
"""Risk-return sensitivity for the adaptive+circuit-breaker bull strategy.

Key question for "make more money without blowing up": where does scaling
per-trade risk become dangerous? The 2024-26 bull era scales beautifully
(PF ~10.8, maxDD stays <8% even at x10 risk), but the FULL 2004-26 history
shows maxDD jumping to 16.5% at only x2 — because weak regimes (chop/high-vol)
trade poorly. This script breaks risk scaling down by YEAR to find the danger
zones, and tests a regime-asymmetric risk allocation (more risk in strong
regimes, less in weak) as a better lever than uniform scaling.

Usage: py scripts/risk_elasticity_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.bull_adaptive import AdaptiveConfig, run_adaptive

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)

YEARS = [("2019", "2019-01-01", "2019-12-31"), ("2020", "2020-01-01", "2020-12-31"),
         ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
         ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-28")]


def stats(cfg, mask, maxdd_mc=None):
    era = D[mask].reset_index(drop=True)
    res = run_adaptive(era, cfg)
    st = res["stats"]
    mc = maxdd_mc if maxdd_mc is not None else cfg.margin_call_pct * 100
    blew = st["max_drawdown"] >= mc - 0.5
    return st, blew


def report(cfg, mask, label):
    st, blew = stats(cfg, mask)
    return f"{label:>16} | ret{st['total_return']:+7.1f}% PF{st['profit_factor']:5.2f} " \
           f"maxDD{st['max_drawdown']:5.1f}% win{st['win_rate']:3.0f}% tr{st['trade_count']:3d} " \
           f"{'[BLOWUP]' if blew else 'safe'}"


def mk(mult):
    return AdaptiveConfig(risk_10x=0.005 * mult, risk_5x=0.01 * mult, risk_low=0.02 * mult)


MASK_FULL = pd.Series(True, index=D.index)
ERA = (D["date"] >= "2024-01-01") & (D["date"] <= "2026-08-28")
BULL_ERA = ERA

print("=== 逐年 · 单笔风险放大 (x1=基线, x3, x6) ===")
print("-" * 80)
hdr = f"{'year':>6}" + "".join(f" | x{m:<18}" for m in [1.0, 3.0, 6.0])
print(hdr)
print("-" * 80)
for yname, s, e in YEARS:
    mask = (D["date"] >= s) & (D["date"] <= e)
    row = f"{yname:>6}"
    for m in [1.0, 3.0, 6.0]:
        row += f" | {report(mk(m), mask, f'x{m}')}"
    print(row)

print()
print("=== 区间总览: 平均收益 / 平均回撤 / 是否爆仓 ===")
print("-" * 80)
for label, mask in [("bull-era 2024-26", BULL_ERA), ("full 2004-26", MASK_FULL)]:
    for m in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        try:
            print(report(mk(m), mask, f"{label} x{m}"))
        except Exception as e:
            print(f"{label} x{m}: ERR {type(e).__name__}")
