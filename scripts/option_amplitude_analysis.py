# -*- coding: utf-8 -*-
"""Amplitude x Direction analysis for the NONLINEAR option-buyer framework.

An option buyer needs (1) clear direction AND (2) big enough amplitude (real vol
exceeding the premium). This quantifies, on gold daily data, how often the market
offers BOTH at once vs being a low-amplitude/choppy regime (option-killer).

It reports, by period, the vol percentile (ATR% rolling rank), direction
(bull + rhythm trend), and the share of days that are 'option-buyable' =
high-amplitude AND directional. Also the expected move over the next N days from
today's ATR (a proxy for 'will the move clear the breakeven premium?').

Usage: py scripts/option_amplitude_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.indicators import atr, market_state  # noqa: E402

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)

close = D["close"]
high = D["high"]
low = D["low"]
D["atr14"] = atr(high, low, close, 14)
D["atr_pct"] = D["atr14"] / close * 100.0
D["vol_pctl"] = D["atr_pct"].rolling(250, min_periods=120).rank(pct=True)

ms = market_state(close, high, low, er_thr=0.10, adx_thr=16.0, chop_hi=68.0)
D["dir"] = ms["dir"].to_numpy()
D["state"] = ms["state"].to_numpy()

# option-buyable: high amplitude (vol_pctl>=0.6) AND directional (up or down trend)
D["option_ok"] = ((D["vol_pctl"] >= 0.60) & (D["state"] == "trend")).astype(float)

WINDOWS = [("2019", "2019-01-01", "2019-12-31"), ("2020", "2020-01-01", "2020-12-31"),
           ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
           ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
           ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-28")]

print("=== 幅度×方向 分析 (期权买方视角) ===")
print("%5s | %7s %7s | %7s %7s | %7s %7s | %8s" % (
    "year", "volPctl", "atr%", "up_trend", "dn_trend", "chop", "option_ok", "check"))
print("-" * 92)
for y, s, e in WINDOWS:
    yy = D[(D["date"] >= s) & (D["date"] <= e)]
    vol = yy["vol_pctl"].mean()
    atrp = yy["atr_pct"].mean()
    up = (yy["state"] == "trend") & (yy["dir"] > 0)
    dn = (yy["state"] == "trend") & (yy["dir"] < 0)
    chop = yy["state"] == "chop"
    ok = yy["option_ok"]
    print("%5s | %7.2f %7.2f | %7.1f%% %7.1f%% | %7.1f%% %7.1f%% | %8s" % (
        y, vol, atrp, up.mean() * 100, dn.mean() * 100, chop.mean() * 100, ok.mean() * 100,
        "可买期权" if ok.mean() >= 0.2 else "较少/震荡"))

# today's expected N-day move (proxy for breakeven-swing feasibility)
print("\n=== 最近一天: 波动率分位 / 未来N日预期波动幅度 ===")
last = D.iloc[-1]
print("date=%s close=%.1f atr14=%.1f (%.2f%%) vol_pctl=%.2f state=%s dir=%s" % (
    last["date"].date(), last["close"], last["atr14"], last["atr_pct"], last["vol_pctl"],
    last["state"], last["dir"]))
for n in [5, 10, 20, 30]:
    # expected move over n days ~ ATR * sqrt(n)
    exp_move = last["atr14"] * np.sqrt(n)
    print("  %2d日预期波动幅度 ≈ ±%.1f (±%.1f%%)" % (n, exp_move, exp_move / last["close"] * 100))
