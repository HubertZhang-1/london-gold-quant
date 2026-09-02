# -*- coding: utf-8 -*-
"""Adaptive leverage + circuit breaker upgrades to survive extreme years.

Diagnosis: 2020/2023/2026 blow up via margin_call because:
  - risk_per_trade_pct=20% too high (a single stop eats 20%; 2 stops -> halt)
  - leverage high in a single-direction (long) trend that reverses.
Fixes (multi-layer):
  (a) risk_per_trade_pct lowered to 0.5-1% so single-trade loss is bounded
  (b) extreme-volatility flat rule: atr_pctl > 0.85 -> leverage 0 (stand aside)
  (c) earlier margin_call (e.g. peak drawdown > 15% -> flat+stop)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import atr, ema, efficiency_ratio, trend_regime
from london_gold.leverage_backtest import run_leverage_backtest

D = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1d.csv", parse_dates=["date"])
D["date"] = pd.to_datetime(D["date"], utc=True)
D = D.sort_values("date").reset_index(drop=True)
close = D["close"]; high = D["high"]; low = D["low"]
atr14 = atr(high, low, close, 14)
atr_pct = atr14 / close * 100.0
atr_pctl = atr_pct.rolling(250, min_periods=120).rank(pct=True)
er20 = efficiency_ratio(close, 20)
ema200 = ema(close, 200); ema_slope = ema200.diff(20)
up_trend = (close > ema(close, 50)).astype(float)
trend = trend_regime(close, high, low, er_window=20, er_threshold=0.12, adx_window=14, adx_threshold=20)
bull = np.clip((ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3, 0, 1)
D["bull"] = bull; D["er20"] = er20; D["atr_pctl"] = atr_pctl


def lev_for(row, ext_flat=True):
    b = row["bull"]; er = row["er20"] if not np.isnan(row["er20"]) else 0
    vol = row["atr_pctl"] if not np.isnan(row["atr_pctl"]) else 0.5
    if b < 0.55: return 0.0
    if ext_flat and vol > 0.85: return 0.0        # extreme volatility -> stand aside
    if vol > 0.80: return 1.0
    if er > 0.25 and vol < 0.65: return 10.0
    if er > 0.15: return 5.0
    return 2.0


D["lev"] = D.apply(lambda r: lev_for(r), axis=1)
MICRO_W = {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
           "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4}

WINDOWS = [("2019", "2019-01-01", "2019-12-31"), ("2020", "2020-01-01", "2020-12-31"),
           ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
           ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
           ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-08-28")]


def run(part, risk, mc):
    fac = build_factors(part)
    micro = aggregate_score(fac, MICRO_W)
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=20,
                       er_threshold=0.12, adx_window=14, adx_threshold=20).to_numpy()
    sig = np.where((reg > 0.5) & (micro > 0.5), 1, np.where((reg > 0.5) & (micro < -0.5), -1, 0))
    sig = np.where(part["bull"].to_numpy() > 0.55, sig, 0)
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy(); anim = ~np.isnan(a)
    f = pd.DataFrame({"date": part["date"], "open": part["open"], "high": part["high"],
                      "low": part["low"], "close": part["close"], "signal": sig,
                      "stop_dist": np.where(anim, a * 2.0, 0.0), "tp_dist": np.where(anim, a * 2.0 * 2.0, 0.0)})
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, leverage=3.0, risk_per_trade_pct=risk,
                      margin_call_pct=mc)
    return run_leverage_backtest(f, cost, "x", leverage_series=np.clip(part["lev"].to_numpy(), 0, None))["stats"]


print("=== ADAPTIVE + CIRCUIT BREAKER (risk 0.5%/1%, mc 15%/20%) ===")
print(f"{'year':>5} | {'r0.5/mc15':>10} {'r1/mc15':>9} {'r0.5/mc20':>10} {'r1/mc20':>9}")
print("-" * 56)
for wname, s, e in WINDOWS:
    part = D[(D["date"] >= s) & (D["date"] <= e)].reset_index(drop=True)
    a = run(part, 0.005, 0.15)
    b = run(part, 0.01, 0.15)
    c = run(part, 0.005, 0.20)
    d = run(part, 0.01, 0.20)
    print(f"{wname:>5} | {a['total_return']:+10.1f} {b['total_return']:+9.1f} {c['total_return']:+10.1f} {d['total_return']:+9.1f}")
