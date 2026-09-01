# -*- coding: utf-8 -*-
"""Check available XAUUSD 5m datasets and their coverage, aggregate to 1h."""
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\张策\Documents\EA量化项目\data")

for name in ("XAUUSD_5m.csv", "XAUUSD_5m_2026.csv", "XAUUSD_5m_raw.jsonl"):
    p = BASE / name
    if p.suffix == ".jsonl":
        continue
    if not p.exists():
        print(name, "MISSING")
        continue
    df = pd.read_csv(p)
    if "date" not in df.columns:
        print(name, "no date col", list(df.columns)[:6])
        continue
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"{name:24s} rows={len(df):7d}  {df['date'].min()}  ->  {df['date'].max()}")

    # aggregate to 1h for rolling oos
    if {"open", "high", "low", "close"}.issubset(df.columns):
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        h1 = df.set_index("date").resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna().reset_index()
        out = BASE / f"{name.replace('.csv', '')}_1h.csv"
        h1.to_csv(out, index=False)
        print(f"    -> 1h: rows={len(h1)}  {h1['date'].min()} -> {h1['date'].max()}  saved {out.name}")
