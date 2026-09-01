# -*- coding: utf-8 -*-
"""Diagnose why the curated factor strategy failed on 2018-2023 vs 2024-2026.
Compare regime indicators and per-window factor behavior between periods."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.factor_library import build_factors, aggregate_score
from london_gold.indicators import adx, atr, efficiency_ratio

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)

periods = [
    ("2018H1", "2018-01-01", "2018-06-30"),
    ("2018H2", "2018-07-01", "2018-12-31"),
    ("2019H1", "2019-01-01", "2019-06-30"),
    ("2019H2", "2019-07-01", "2019-12-31"),
    ("2020H1", "2020-01-01", "2020-06-30"),
    ("2020H2", "2020-07-01", "2020-12-31"),
    ("2021H1", "2021-01-01", "2021-06-30"),
    ("2021H2", "2021-07-01", "2021-12-31"),
    ("2022H1", "2022-01-01", "2022-06-30"),
    ("2022H2", "2022-07-01", "2022-12-31"),
    ("2023H1", "2023-01-01", "2023-06-30"),
    ("2023H2", "2023-07-01", "2023-12-31"),
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
    ("2026H1", "2026-01-01", "2026-06-30"),
]

print(f"{'period':>8} {'ADX_m':>6} {'ER_m':>6} {'ATR_m':>6} {'ADR%':>6} {'win%?':>6}")
print("-" * 50)
for pname, s, e in periods:
    part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
    if len(part) < 50:
        print(f"{pname:>8} (insufficient)")
        continue
    adx_m = adx(part["high"], part["low"], part["close"], 14).mean()
    er_m = efficiency_ratio(part["close"], 48).mean()
    atr_m = atr(part["high"], part["low"], part["close"], 14).mean()
    # annualized daily range %
    d_range = (part["high"] - part["low"]).mean() / part["close"].mean() * 100
    print(f"{pname:>8} {adx_m:6.1f} {er_m:6.3f} {atr_m:6.2f} {d_range:6.2f}")

print()
print("Interpretation: if 2018-23 has similar ADX/ER as 2024-26 but the strategy")
print("still fails, the edge is regime-independent and likely overfit. If ADX/ER")
print("is much lower in 2018-23, gold was choppier and momentum naturally fails.")
