# -*- coding: utf-8 -*-
"""Convert the XAUUSD 5m jsonl dataset into the engine's CSV format.

Input : data/XAUUSD_5m_raw.jsonl   (one JSON object per line)
Output: data/XAUUSD_5m.csv         (date,open,high,low,close[,volume])
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def convert(
    raw_path: Path,
    out_path: Path,
    timezone: str | None = None,
) -> pd.DataFrame:
    rows = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
    if timezone:
        df["date"] = df["date"].dt.tz_localize(timezone).dt.tz_convert("UTC").dt.tz_localize(None)
    for name in ("Open", "High", "Low", "Close"):
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    invalid = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )
    if invalid.any():
        print(f"dropped {int(invalid.sum())} invalid OHLC rows")
        df = df[~invalid].reset_index(drop=True)
    columns = ["date", "open", "high", "low", "close"]
    if "Volume" in df.columns:
        df["volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        columns.append("volume")
    df[columns].to_csv(out_path, index=False, encoding="utf-8")
    return df[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XAUUSD 5m jsonl to engine CSV")
    parser.add_argument("--raw", default=str(PROJECT_ROOT / "data" / "XAUUSD_5m_raw.jsonl"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "XAUUSD_5m.csv"))
    parser.add_argument("--timezone", default=None, help="e.g. Etc/GMT-2 if the jsonl is in broker server time")
    args = parser.parse_args()

    df = convert(Path(args.raw), Path(args.out), args.timezone)
    print(f"rows={len(df)} first={df['date'].iloc[0]} last={df['date'].iloc[-1]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
