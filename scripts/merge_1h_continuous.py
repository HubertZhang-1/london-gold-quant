# -*- coding: utf-8 -*-
"""Merge all 5m XAUUSD sources into a continuous 1h frame for rolling OOS.

Sources (5m): long history (2004..2025-07), 2025H2 (2025-08..12, HistData),
2026 (Jan..Aug). Each is aggregated to 1h, deduped by date, concatenated.
"""
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\张策\Documents\EA量化项目\data")
SYMBOL = "XAUUSD"


def agg_1h(df5: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a 5m OHLC frame to 1h."""
    df = df5.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("date")
    h1 = df.set_index("date").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna().reset_index()
    return h1


# raw 5m sources (files may be .csv or the long-history csv)
srcs = [BASE / "XAUUSD_5m.csv"]  # 2004..2025-07/09
if (BASE / "XAUUSD_5m_2025h2.csv").exists():
    srcs.append(BASE / "XAUUSD_5m_2025h2.csv")  # 2025-08..12
if (BASE / "XAUUSD_5m_2026.csv").exists():
    srcs.append(BASE / "XAUUSD_5m_2026.csv")  # 2026-01..08

frames = []
for s in srcs:
    if not s.exists():
        print(f"skip missing {s.name}")
        continue
    print(f"aggregating {s.name}")
    frames.append(agg_1h(pd.read_csv(s)))

merged = pd.concat(frames, ignore_index=True)
merged = merged.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
out = BASE / "XAUUSD_1h_continuous.csv"
merged.to_csv(out, index=False)
print(f"merged 1h: rows={len(merged)}  {merged['date'].min()} -> {merged['date'].max()}")
print(f"saved {out.name}")
