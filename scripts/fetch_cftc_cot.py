# -*- coding: utf-8 -*-
"""Resilient CFTC disaggregated COT downloader for COMEX GOLD.

CFTC rate-limits direct requests (403 under burst). We fetch one year at a
time with delays + retries, and save incremental results. Only GOLD rows are
kept. Run with the cache so repeated runs resume instead of re-fetching.
"""
from __future__ import annotations

import csv
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = Path(r"C:\Users\张策\Documents\EA量化项目\data")
H = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
YEARS = list(range(2018, 2027))
SINGLE_YEAR = sys.argv[1] if len(sys.argv) > 1 else None


def fetch_year(yr: int) -> pd.DataFrame:
    tmpls = (
        "https://www.cftc.gov/dea/history/fut_disagg_txt_{y}.zip",
        "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{y}.zip",
    )
    for attempt in range(3):
        for tmpl in tmpls:
            url = tmpl.format(y=yr)
            try:
                r = requests.get(url, headers=H, timeout=90)
            except requests.RequestException:
                continue
            if r.status_code != 200 or r.content[:2] != b"PK":
                continue
            try:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                text = z.read(z.namelist()[0]).decode("latin-1")
            except zipfile.BadZipFile:
                continue
            rows = list(csv.DictReader(io.StringIO(text)))
            df = pd.DataFrame(rows)
            # find GOLD rows by any column containing 'GOLD'
            gold = df[df.apply(lambda r: any("GOLD" in str(v).upper() for v in r.values), axis=1)]
            return gold
        time.sleep(8 + attempt * 5)
    return pd.DataFrame()


def main() -> None:
    frames = []
    years = [SINGLE_YEAR] if SINGLE_YEAR else YEARS
    for yr in years:
        yr = int(yr)
        cache = BASE / f"cftc_gold_disagg_{yr}.csv"
        if cache.exists():
            g = pd.read_csv(cache)
            print(f"  {yr}: cached {len(g)} rows")
            frames.append(g)
            continue
        g = fetch_year(yr)
        if g.empty:
            print(f"  {yr}: FAILED (rate-limited)")
            time.sleep(10)
            continue
        # normalize date col name(s)
        datecol = [c for c in g.columns if "Report_Date" in c]
        if not datecol:
            print(f"  {yr}: no date col")
            continue
        g = g.rename(columns={datecol[0]: "date"})
        g["date"] = pd.to_datetime(g["date"], errors="coerce")
        g = g.dropna(subset=["date"])
        g.to_csv(cache, index=False, encoding="utf-8")
        print(f"  {yr}: saved {len(g)} GOLD rows -> {cache.name}")
        frames.append(g)
        time.sleep(6)

    # build unified dataset with the five categories (match actual CFTC columns)
    cols = {
        "Prod_Merc_Positions_Long_All": "prod_long",
        "Prod_Merc_Positions_Short_All": "prod_short",
        "Swap_Positions_Long_All": "swap_long",
        "Swap__Positions_Short_All": "swap_short",
        "Swap_Positions_Short_All": "swap_short",
        "M_Money_Positions_Long_All": "mgr_long",
        "M_Money_Positions_Short_All": "mgr_short",
        "Other_Rept_Positions_Long_All": "oth_long",
        "Other_Rept_Positions_Short_All": "oth_short",
        "NonRept_Positions_Long_All": "nonrept_long",
        "NonRept_Positions_Short_All": "nonrept_short",
        "Conc_Gross_LE_4_TDR_Short_All": "conc4_short",
    }
    uni = []
    for g in frames:
        present = {k: v for k, v in cols.items() if k in g.columns}
        if not present:
            print(f"  warn: no category cols in a cached year")
            continue
        sub = g[["date"] + list(present)].rename(columns=present).copy()
        sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
        sub = sub.dropna(subset=["date"])
        for c in set(present.values()) - {"date"}:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.groupby("date", as_index=False).first()
        uni.append(sub)
    if not uni:
        print("no COT data saved")
        return
    cot = pd.concat(uni, ignore_index=True).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    out = BASE / "gold_cot_disagg.csv"
    cot.to_csv(out, index=False, encoding="utf-8")
    print(f"\nsaved {out.name}: {len(cot)} rows, {cot['date'].min()} -> {cot['date'].max()}")
    print("cols:", list(cot.columns))


if __name__ == "__main__":
    main()
