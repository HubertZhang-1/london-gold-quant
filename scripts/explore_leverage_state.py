# -*- coding: utf-8 -*-
"""Explore a market-state classifier that drives adaptive leverage.

State features (daily):
  - volatility percentile (ATR % of price vs its rolling history)  -> high = cap leverage
  - efficiency ratio (trend cleanliness)                          -> high = allow higher leverage
  - bull score (long-trend up + up-trend)                        -> bull = participate
Map state -> leverage: bear=0, high-vol=1-2x, normal bull=3-5x, clean trend=10x.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.indicators import atr, ema, efficiency_ratio, trend_regime

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)
close = D["close"]
high = D["high"]
low = D["low"]

atr14 = atr(high, low, close, 14)
atr_pct = atr14 / close * 100.0
atr_pctl = atr_pct.rolling(250, min_periods=120).rank(pct=True)  # vol percentile vs 1y
er20 = efficiency_ratio(close, 20)
ema200 = ema(close, 200)
ema_slope = ema200.diff(20)
up_trend = (close > ema(close, 50)).astype(float)
trend = trend_regime(close, high, low, er_window=20, er_threshold=0.12, adx_window=14, adx_threshold=20)
bull = np.clip((ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3, 0, 1)

D["atr_pctl"] = atr_pctl
D["er20"] = er20
D["bull"] = bull
D["atr_pct"] = atr_pct


def state_label(row):
    """Return state + suggested leverage."""
    b = row["bull"]
    er = row["er20"] if not np.isnan(row["er20"]) else 0
    vol = row["atr_pctl"] if not np.isnan(row["atr_pctl"]) else 0.5
    if b < 0.55:
        return ("BEAR", 0)
    if vol > 0.80:
        return ("HIGH_VOL", 1.0)
    if er > 0.25 and vol < 0.65:
        return ("CLEAN_TREND", 10.0)
    if er > 0.15:
        return ("BULL", 5.0)
    return ("CHOP", 2.0)


D["state"], D["leverage"] = zip(*D.apply(state_label, axis=1))

# Summarize per year: dominant state and avg suggested leverage
print("=== YEARLY market-state summary (daily) ===")
print(f"{'year':>5} {'bull%':>5} {'er':>5} {'volP':>5} | {'dominant':>6} {'avgLev':>6}")
print("-" * 46)
for y in range(2018, 2027):
    seg = D[(D["date"] >= f"{y}-01-01") & (D["date"] <= f"{y}-12-31")]
    if len(seg) < 30:
        continue
    dom = seg["state"].value_counts().idxmax()
    avgLev = seg["leverage"].mean()
    print(f"{y:>5} {seg['bull'].mean()*100:5.1f} {seg['er20'].mean():5.2f} {seg['atr_pctl'].mean():5.2f} | "
          f"{dom:>6} {avgLev:6.1f}")
