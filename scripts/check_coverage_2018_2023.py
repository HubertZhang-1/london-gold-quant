# -*- coding: utf-8 -*-
"""Check data coverage for 2018-2023 independent OOS validation."""
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\张策\Documents\EA量化项目\data")
df = pd.read_csv(BASE / "XAUUSD_1h_continuous.csv")
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values("date").reset_index(drop=True)
print(f"continuous 1h: {df['date'].min()} -> {df['date'].max()}, rows={len(df)}")
for y in range(2018, 2027):
    seg = df[(df["date"] >= f"{y}-01-01") & (df["date"] <= f"{y}-12-31")]
    if len(seg) == 0:
        print(f"  {y}: NO DATA")
    else:
        print(f"  {y}: rows={len(seg):6d}  {seg['date'].min().date()} -> {seg['date'].max().date()}")
