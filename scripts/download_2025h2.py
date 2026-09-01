# -*- coding: utf-8 -*-
"""Download 2025-08 .. 2025-12 XAUUSD M1 from HistData, aggregate to 5m."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from download_histdata_xauusd import download_month  # noqa: E402

BASE = Path(r"C:\Users\张策\Documents\EA量化项目\data")


def main() -> None:
    year = 2025
    months = [8, 9, 10, 11, 12]
    frames = []
    for month in months:
        try:
            df = download_month(year, month)  # returns 1m OHLC frame
            print(f"downloaded {year}-{month:02d}: rows={len(df)}")
            if len(df):
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {year}-{month:02d}: {exc}")
        time.sleep(2)

    if not frames:
        print("no data downloaded")
        return

    # concat 1m, aggregate to 5m
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    resampled = (
        merged.set_index("date")
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    out = BASE / "XAUUSD_5m_2025h2.csv"
    resampled.to_csv(out, index=False, encoding="utf-8")
    print(f"\nmerged 1m rows={len(merged)}; 5m rows={len(resampled)}")
    print(f"range {resampled['date'].min()} -> {resampled['date'].max()}")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
