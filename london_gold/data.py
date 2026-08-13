# -*- coding: utf-8 -*-
"""London gold (XAU) daily history and realtime quotes via AkShare."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "data" / "london_gold_daily.csv"
DEFAULT_SCAN_FILE = PROJECT_ROOT / "data" / "london_gold_scan.json"


def load_daily(cache_path: str | Path = DEFAULT_CACHE) -> pd.DataFrame | None:
    """Load cached daily bars, or None when the cache file is missing."""
    cache = Path(cache_path)
    if not cache.exists():
        return None
    df = pd.read_csv(cache, parse_dates=["date"])
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_daily(
    force: bool = False,
    cache_path: str | Path = DEFAULT_CACHE,
    max_cache_age_days: int = 2,
) -> pd.DataFrame:
    """Download XAU daily bars, cache locally and reuse fresh cache."""
    cache = Path(cache_path)
    if not force and cache.exists():
        df = load_daily(cache)
        if df is not None and len(df) > 0:
            last = df["date"].max().date()
            today = datetime.now().date()
            if (today - last).days <= max_cache_age_days:
                return df

    import akshare as ak

    raw = ak.futures_foreign_hist(symbol="XAU")
    if raw is None or raw.empty:
        raise RuntimeError("akshare returned no XAU daily data")

    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "open", "high", "low", "close"]].copy()

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def fetch_realtime() -> dict:
    """Fetch the latest London gold quote from the Sina commodity feed."""
    import akshare as ak

    raw = ak.futures_foreign_commodity_realtime(symbol=["XAU"])
    if raw is None or raw.empty:
        raise RuntimeError("akshare returned no XAU realtime quote")
    row = raw.iloc[0]
    return {
        "name": str(row.get("名称", "伦敦金")),
        "last": float(row.get("最新价", 0.0) or 0.0),
        "change": float(row.get("涨跌额", 0.0) or 0.0),
        "change_pct": float(row.get("涨跌幅", 0.0) or 0.0),
        "open": float(row.get("开盘价", 0.0) or 0.0),
        "high": float(row.get("最高价", 0.0) or 0.0),
        "low": float(row.get("最低价", 0.0) or 0.0),
        "bid": float(row.get("买价", 0.0) or 0.0),
        "ask": float(row.get("卖价", 0.0) or 0.0),
        "time": str(row.get("行情时间", "")),
        "date": str(row.get("日期", "")),
    }
