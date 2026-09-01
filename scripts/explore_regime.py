# -*- coding: utf-8 -*-
"""Explore market-regime indicators across the rolling windows to design a filter."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.indicators import adx, atr, ema

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)

close = DF["close"]
high = DF["high"]
low = DF["low"]

# Regime indicators
adx_h1 = adx(high, low, close, 14)
atr_h1 = atr(high, low, close, 14)
# efficiency ratio over 48h: |close - close[-48]| / sum(|delta|)
close_arr = close.to_numpy(float)
er = pd.Series(np.nan, index=DF.index)
for i in range(48, len(DF)):
    num = abs(close_arr[i] - close_arr[i - 48])
    den = np.abs(np.diff(close_arr[i - 48:i + 1])).sum()
    er.iloc[i] = num / den if den > 0 else 0.0
# 20-day ATR percentile (longer horizon)
atr_20d = atr(high, low, close, 20 * 24)
atr_pct = atr_h1.rolling(20 * 24).rank(pct=True)

DF["adx_h1"] = adx_h1
DF["er48"] = er
DF["atr_pct20"] = atr_pct
DF["adx_pct"] = adx_h1.rolling(20 * 24).rank(pct=True)

# Window summary
windows = [
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
    ("2025H2", "2025-07-01", "2025-12-31"),
    ("2026H1", "2026-01-01", "2026-06-30"),
    ("2026H2", "2026-06-01", "2026-08-28"),
]

print(f"{'window':>8} {'ADX_m':>6} {'ADX_p50':>8} {'ER_m':>6} {'ATR_p50':>8} {'ADX>25%':>8}")
print("-" * 58)
for wname, s, e in windows:
    part = DF[(DF["date"] >= s) & (DF["date"] <= e)]
    if len(part) < 50:
        print(f"{wname:>8} {len(part):6d} (insufficient)")
        continue
    adxm = part["adx_h1"].mean()
    adxp = part["adx_pct"].median()
    erm = part["er48"].mean()
    atrp = part["atr_pct20"].median()
    adxgt25 = (part["adx_h1"] > 25).mean() * 100
    print(f"{wname:>8} {adxm:6.1f} {adxp:8.2f} {erm:6.3f} {atrp:8.2f} {adxgt25:7.1f}%")
