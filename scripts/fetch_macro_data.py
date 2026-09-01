# -*- coding: utf-8 -*-
"""Download macro series (DXY, VIX, 10y nominal yield) from Yahoo, daily, merged.

These form the MACRO layer of the three-line system:
  - DXY  : USD strength (negative driver for gold)
  - VIX  : crisis/volatility gauge (conditional driver)
  - TNX  : 10y nominal yield proxy for real-rate direction (negative driver)

We save a daily CSV that can be forward-filled onto gold 1h bars.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

BASE = Path(r"C:\Users\张策\Documents\EA量化项目\data")
H = {"User-Agent": "Mozilla/5.0"}
SYMBOLS = {
    "dxy": "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?interval=1d&range=9y",
    "vix": "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=9y",
    "tnx": "https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?interval=1d&range=9y",
}


def fetch(symbol: str, url: str) -> pd.DataFrame:
    r = requests.get(url, headers=H, timeout=40)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    quotes = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s", utc=True).normalize(),
        symbol: quotes["close"],
    })
    df = df.dropna(subset=[symbol]).drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def main() -> None:
    parts = []
    for name, url in SYMBOLS.items():
        try:
            df = fetch(name, url)
            print(f"  {name}: {len(df)} rows  {df['date'].min().date()} -> {df['date'].max().date()}")
            parts.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} ERR: {exc}")

    if not parts:
        print("no macro data")
        return
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    out = BASE / "macro_daily.csv"
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"\nsaved {out.name}: {len(merged)} rows, {merged['date'].min()} -> {merged['date'].max()}")
    print("cols:", list(merged.columns))


if __name__ == "__main__":
    main()
