# -*- coding: utf-8 -*-
"""Download several months of XAUUSD M1 and merge+resample to 5m."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from download_histdata_xauusd import download_month  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch download XAUUSD M1 and build 5m CSV")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--months", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "XAUUSD_5m_2026.csv"))
    args = parser.parse_args()

    frames = []
    for month in args.months:
        try:
            df = download_month(args.year, month)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {args.year}-{month:02d}: {exc}")
            time.sleep(2)
            continue
        if len(df):
            frames.append(df)
        time.sleep(2)

    if not frames:
        raise SystemExit("no data downloaded")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

    resampled = (
        merged.set_index("date")
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_csv(out, index=False, encoding="utf-8")
    print(f"merged M1 rows: {len(merged)}")
    print(f"5m rows: {len(resampled)}  first={resampled['date'].iloc[0]}  last={resampled['date'].iloc[-1]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
