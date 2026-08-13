# -*- coding: utf-8 -*-
"""Intraday OHLC for COMEX gold (GC=F), a free proxy for London gold."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "data" / "gc_h1.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def load_intraday(cache_path: str | Path = DEFAULT_CACHE) -> pd.DataFrame | None:
    cache = Path(cache_path)
    if not cache.exists():
        return None
    df = pd.read_csv(cache, parse_dates=["date"])
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_intraday(
    symbol: str = "GC=F",
    interval: str = "1h",
    period: str = "2y",
    force: bool = False,
    cache_path: str | Path = DEFAULT_CACHE,
    max_cache_age_hours: int = 6,
) -> pd.DataFrame:
    """Download Yahoo Finance OHLC bars, cached as naive UTC timestamps."""
    cache = Path(cache_path)
    if not force and cache.exists():
        df = load_intraday(cache)
        if df is not None and len(df) > 0:
            age_hours = (datetime.utcnow() - df["date"].iloc[-1].to_pydatetime()).total_seconds() / 3600
            if age_hours <= max_cache_age_hours:
                return df

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    resp = requests.get(
        url,
        params={"interval": interval, "range": period},
        headers=_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    result = (payload.get("chart") or {}).get("result")
    if not result:
        raise RuntimeError(f"Yahoo returned no data for {symbol}")
    res = result[0]
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    timestamps = res.get("timestamp") or []
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
        }
    )
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df
