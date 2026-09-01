# -*- coding: utf-8 -*-
"""Download XAUUSD minute data from HistData.com (no registration required).

HistData serves one zip per calendar month via POST to /get.php.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fetch_month(year: int, month: int, platform: str = "ASCII") -> requests.Response:
    page_url = (
        f"https://www.histdata.com/download-free-forex-data/?/ascii/"
        f"1-minute-bar-quotes/xauusd/{year}/{month}"
    )
    page = requests.get(page_url, headers=HEADERS, timeout=30)
    page.raise_for_status()
    form = re.search(r'<form[^>]*action="/get\.php"[^>]*>.*?</form>', page.text, re.S)
    if not form:
        raise RuntimeError(f"no download form on {page_url}")
    token = re.search(r'<input[^>]*name="tk"[^>]*value="([^"]*)"', form.group(0))
    if not token:
        raise RuntimeError("no tk token in download form")
    payload = {
        "tk": token.group(1),
        "date": str(year),
        "datemonth": f"{year}{month:02d}",
        "platform": platform,
        "timeframe": "M1",
        "fxpair": "XAUUSD",
    }
    post_headers = dict(HEADERS)
    post_headers["Referer"] = page_url
    resp = requests.post(
        "https://www.histdata.com/get.php",
        data=payload,
        headers=post_headers,
        timeout=120,
        allow_redirects=True,
    )
    resp.raise_for_status()
    if "text/html" in resp.headers.get("Content-Type", ""):
        snippet = resp.text[:200].replace("\n", " ")
        raise RuntimeError(f"download returned html (not a zip) for {year}-{month:02d}: {snippet}")
    return resp


def parse_ascii_bytes(raw: bytes) -> pd.DataFrame:
    """HistData ASCII M1 rows: 'YYYYMMDD HHMMSS;O;H;L;C;V'."""
    text = raw.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or ";" not in line:
            continue
        parts = line.split(";")
        if len(parts) < 6:
            continue
        datetime_str = parts[0]
        try:
            ts = pd.to_datetime(datetime_str, format="%Y%m%d %H%M%S")
        except ValueError:
            continue
        rows.append(
            {
                "date": ts,
                "open": float(parts[1]),
                "high": float(parts[2]),
                "low": float(parts[3]),
                "close": float(parts[4]),
                "volume": int(float(parts[5])) if len(parts) > 5 else 0,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return df


def download_month(
    year: int,
    month: int,
    out_csv: Path | None = None,
) -> pd.DataFrame:
    resp = fetch_month(year, month)
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = next(name for name in zf.namelist() if name.lower().endswith(".csv"))
    raw = zf.read(csv_name)
    df = parse_ascii_bytes(raw)
    print(f"{year}-{month:02d}: rows={len(df)} first={df['date'].iloc[0] if len(df) else '-'} "
          f"last={df['date'].iloc[-1] if len(df) else '-'}")
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"wrote {out_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download XAUUSD M1 from HistData.com")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--out", default=None, help="output csv path (default: data/XAUUSD_1m_YYYYMM.csv)")
    args = parser.parse_args()
    out = Path(args.out) if args.out else PROJECT_ROOT / "data" / f"XAUUSD_1m_{args.year}{args.month:02d}.csv"
    download_month(args.year, args.month, out)


if __name__ == "__main__":
    main()
