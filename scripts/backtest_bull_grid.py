# -*- coding: utf-8 -*-
"""Backtest the bull-only grid WITH the market-rhythm (choppiness) gate.

The rhythm identifier (indicators.market_state) only allows longs in a confirmed
UP trend and stands aside in chop/range. Default gate is the balanced setting:
  chop_hi=68, er_thr=0.10, adx_thr=16  -> keeps most bull-era upside, filters chop.

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
from london_gold.indicators import market_state  # noqa: E402

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)

MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
MACRO["date"] = pd.to_datetime(MACRO["date"], utc=True)
MACRO = MACRO.sort_values("date").reset_index(drop=True)
macro_on_bars = forward_fill_macro(macro_direction_score(MACRO)["macro_score"],
                                   D["date"].to_numpy()).to_numpy()

cfg = BullGridConfig(risk_per_trade_pct=0.02)  # risk-budgeted sizing + balanced rhythm gate
ERA = (D["date"] >= "2024-01-01") & (D["date"] <= "2026-08-28")
FULL = (D["date"] >= "2019-01-01") & (D["date"] <= "2026-08-28")


def run(mask, c):
    sub = D[mask].reset_index(drop=True)
    return run_bull_grid_backtest(sub, c, macro_series=macro_on_bars[mask.to_numpy()])["stats"]


def fmt(s):
    return ("ret{:+.1f}% PF{:.2f} maxDD{:.1f}% win{:.0f}% tr{} net${:+,.0f}".format(
        s["total_return_pct"], s["profit_factor"], s["max_drawdown_pct"],
        s["winrate"] * 100, s["trades"], s["net_pnl"]))


print("=" * 74)
print("牛市单向版 + 行情节奏门 (balanced gate chop68/er.10/adx16) — risk2% stop2.5 rr2")
print("=" * 74)
print("2024-2026 (牛市窗口) : " + fmt(run(ERA, cfg)))
print("2019-2026 (完整区间) : " + fmt(run(FULL, cfg)))

# rhythm state distribution in window + market-state diagnostics
ms = market_state(D["close"], D["high"], D["low"], er_thr=0.10, adx_thr=16.0, chop_hi=68.0)
D["st"] = ms["state"].to_numpy()
D["sig"] = ms["signal"].to_numpy()
war = D[ERA]
print("\n窗口内节奏状态分布: trend %.1f%%  chop %.1f%%  neutral %.1f%%" % (
    (war["st"] == "trend").mean() * 100, (war["st"] == "chop").mean() * 100,
    (war["st"] == "neutral").mean() * 100))
print("窗口内 long 信号占比: %.1f%%" % ((war["sig"] == 1).mean() * 100))

print("\n逐年:")
WINDOWS = [("2019", "2019-01-01", "2019-12-31"), ("2020", "2020-01-01", "2020-12-31"),
           ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
           ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
           ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-28")]
for y, s, e in WINDOWS:
    m = (D["date"] >= s) & (D["date"] <= e)
    print("%5s | %s" % (y, fmt(run(m, cfg))))

# save artifacts
out = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "bull_grid_rhythm"
res = run_bull_grid_backtest(D[ERA].reset_index(drop=True), cfg, macro_series=macro_on_bars[ERA.to_numpy()])
res["equity"].to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
res["trades"].to_csv(f"{out}_trades.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved {out}_equity.csv / _trades.csv")
