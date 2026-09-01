# -*- coding: utf-8 -*-
"""Determine the exact coverage gap in the XAUUSD 5m data after 2025-07."""
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\张策\Documents\EA量化项目\data")

# ZombitX64 long history (2004..2025-09, with a gap after 2025-07)
main = pd.read_csv(BASE / "XAUUSD_5m.csv")
main["date"] = pd.to_datetime(main["date"], utc=True)
# HistData 2026 (Jan..Aug)
y26 = pd.read_csv(BASE / "XAUUSD_5m_2026.csv")
y26["date"] = pd.to_datetime(y26["date"], utc=True)

for name, df in (("XAUUSD_5m.csv", main), ("XAUUSD_5m_2026.csv", y26)):
    print(f"{name}: {df['date'].min()} -> {df['date'].max()}  rows={len(df)}")

print()
print("=== monthly bar counts for 2025-06 .. 2026-01 (long-history file) ===")
seg = main[(main["date"] >= "2025-06-01") & (main["date"] <= "2026-01-31")]
seg = seg.copy()
seg["ym"] = seg["date"].dt.strftime("%Y-%m")
print(seg.groupby("ym").size())

print()
print("=> The gap is 2025-08 .. 2025-12 (and likely early 2026-01).")
print("   HistData single-month downloads will fill 2025-08 .. 2025-12.")
