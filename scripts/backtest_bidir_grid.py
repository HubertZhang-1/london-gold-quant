# -*- coding: utf-8 -*-
"""Backtest the BI-DIRECTIONAL XAUUSD CFD strategy (long in bull, short in bear).

The user's instrument is a linear CFD that can go both ways and CAN blow up, so the
correct usage is to trade BOTH directions with the trend (牛市做多 / 熊市做空 /
震荡不参与) rather than a bull-only long. This compares the bidirectional version
against the bull-only version on the 2024-2026 window and the full history.

Usage: py scripts/backtest_bidir_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.bidir_grid import BidirGridConfig, run_bidir_grid_backtest  # noqa: E402
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


def fmt(st):
    return ("ret{:+.1f}% PF{:.2f} maxDD{:.1f}% win{:.0f}% tr{} net${:+,.0f} (L{} S{})".format(
        st["total_return_pct"], st["profit_factor"], st["max_drawdown_pct"],
        st["winrate"] * 100, st["trades"], st["net_pnl"],
        st.get("long_trades", 0), st.get("short_trades", 0)))


bidir = BidirGridConfig(risk_per_trade_pct=0.02, stop_mult=2.5, rr=2.0)
bullonly = BullGridConfig(risk_per_trade_pct=0.02, stop_mult=2.5, rr=2.0)

print("=== 双向(优化做空) vs 只做多 ===")
print("优化做空门: 距60日高点回落≥12% 且 止损后冷却3根才做空")
print("-" * 84)
for label, cfg, runner in [("双向CFD", bidir, run_bidir_grid_backtest),
                           ("只做多", bullonly, run_bull_grid_backtest)]:
    sub = D[ERA].reset_index(drop=True)
    res = runner(sub, cfg, macro_series=macro_on_bars[ERA.to_numpy()])
    print("%-8s | %s" % (label, fmt(res["stats"])))

print("\n=== 完整区间 2019-2026 (含熊市年): 双向 vs 只做多 ===")
for label, cfg, runner in [("双向CFD", bidir, run_bidir_grid_backtest),
                           ("只做多", bullonly, run_bull_grid_backtest)]:
    sub = D[FULL].reset_index(drop=True)
    res = runner(sub, cfg, macro_series=macro_on_bars[FULL.to_numpy()])
    print("%-8s | %s" % (label, fmt(res["stats"])))

print("\n=== 双向 逐年 (risk2% stop2.5 rr2) ===")
WINDOWS = [("2019", "2019-01-01", "2019-12-31"), ("2020", "2020-01-01", "2020-12-31"),
           ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
           ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
           ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-28")]
for y, s, e in WINDOWS:
    m = (D["date"] >= s) & (D["date"] <= e)
    sub = D[m].reset_index(drop=True)
    res = run_bidir_grid_backtest(sub, bidir, macro_series=macro_on_bars[m.to_numpy()])
    print("%5s | %s" % (y, fmt(res["stats"])))

# save artifacts
out = Path(r"C:\Users\张策\Documents\EA量化项目\reports") / "bidir_grid"
sub = D[ERA].reset_index(drop=True)
res = run_bidir_grid_backtest(sub, bidir, macro_series=macro_on_bars[ERA.to_numpy()])
res["equity"].to_csv(f"{out}_equity.csv", index=False, encoding="utf-8-sig")
res["trades"].to_csv(f"{out}_trades.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved {out}_equity.csv / _trades.csv")
