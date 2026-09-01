# -*- coding: utf-8 -*-
"""Merge 2024-2025 1h and 2026 1h into a continuous 1h frame for rolling OOS."""
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\张策\Documents\EA量化项目\data")
main = pd.read_csv(BASE / "XAUUSD_5m_1h.csv")   # 2004-2025-09
y26 = pd.read_csv(BASE / "XAUUSD_5m_2026_1h.csv")  # 2026-01-08
for df in (main, y26):
    df["date"] = pd.to_datetime(df["date"], utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
merged = pd.concat([main, y26], ignore_index=True)
merged = merged.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
out = BASE / "XAUUSD_1h_continuous.csv"
merged.to_csv(out, index=False)
print(f"merged 1h: rows={len(merged)}  {merged['date'].min()} -> {merged['date'].max()}")
print(f"saved {out.name}")
