# -*- coding: utf-8 -*-
"""Explore a bull/bear regime classifier for gold and review all strategies.

Bull definition for 'long-only in bull strategy':
  price trending UP (long-term EMA slope positive) AND strategy edge active
  (trend regime / volatility) AND macro not strongly bearish (USD not surging).

We compute a continuous 'bull_score' on 1h bars and label each historical window
as BULL (tradable) vs BEAR/CHOP (stand aside).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.indicators import adx, atr, ema, efficiency_ratio, trend_regime

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])

close = DF["close"]
high = DF["high"]
low = DF["low"]

# Long-term trend: EMA(4800h ~ 200 days) slope + price>EMA
ema200 = ema(close, 4800)
ema_slope = ema200.diff(120)                # slope over ~5 days
above_long = (close > ema200).astype(float)
# Trend strength
trend = trend_regime(close, high, low, er_window=48, er_threshold=0.12, adx_window=14, adx_threshold=20.0)
# Trend direction positive (up)
adx14 = adx(high, low, close, 14)
ema50 = ema(close, 1200)
up_trend = (close > ema50).astype(float)
# Volatility activity
atr14 = atr(high, low, close, 14)
vol_active = (atr14 > atr14.rolling(4800, min_periods=1200).median() * 0.8).astype(float)

# Bull score combine (long-only tradable conditions)
bull = (ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3
bull = np.clip(bull, 0, 1)

DF["bull_score"] = bull
DF["above_long"] = above_long
DF["up_trend"] = up_trend
DF["trend"] = trend

# Aggregate per window: fraction of bars in bull regime + avg year return
print(f"{'window':>22} {'bull%':>6} {'aboveLong%':>10} {'upTrend%':>9} | {'yearRet':>8} | label")
print("-" * 72)
for wname, s, e in [
    ("2018", "2018-01-01", "2018-12-31"), ("2019", "2019-01-01", "2019-12-31"),
    ("2020", "2020-01-01", "2020-12-31"), ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-08-28"),
]:
    seg = DF[(DF["date"] >= s) & (DF["date"] <= e)]
    if len(seg) < 50:
        continue
    bull_pct = seg["bull_score"].mean() * 100
    abovelong = seg["above_long"].mean() * 100
    uptrend = seg["up_trend"].mean() * 100
    year_ret = (seg["close"].iloc[-1] / seg["close"].iloc[0] - 1) * 100
    label = "BULL" if bull_pct > 55 else ("CHOP" if bull_pct > 35 else "BEAR")
    print(f"{wname:>22} {bull_pct:6.1f} {abovelong:10.1f} {uptrend:9.1f} | {year_ret:+8.1f} | {label}")
