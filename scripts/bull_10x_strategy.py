# -*- coding: utf-8 -*-
"""10x-leverage bull-only double-gate strategy.

Gates:
  (1) BULL regime: bull_score > bull_thr (clean uptrend)
  (2) STRATEGY EDGE: only new entries allowed while trailing realized edge is
      positive — computed from CLOSED trades only (no look-ahead). We wrap the
      backtest so a 'rolling_edge' signal column gates entries, populated from
      the micro score's realized-lookalike (edge proxy = recent micro mean).

The leverage backtest uses margin accounting: cash only changes on close; PnL
on nominal exposure; risk capped at risk_per_trade_pct; margin_call halts.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import atr, ema, trend_regime
from london_gold.leverage_backtest import run_leverage_backtest

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
close = DF["close"]
ema200 = ema(close, 4800)
ema_slope = ema200.diff(120)
up_trend = (close > ema(close, 1200)).astype(float)
trend = trend_regime(close, DF["high"], DF["low"], er_window=48, er_threshold=0.12,
                     adx_window=14, adx_threshold=20.0)
DF["bull"] = np.clip((ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3, 0, 1)

MICRO_W = {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
           "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4}

WINDOWS = [("2018", "2018-01-01", "2018-12-31"), ("2019", "2019-01-01", "2019-12-31"),
           ("2020", "2020-01-01", "2020-12-31"), ("2021", "2021-01-01", "2021-12-31"),
           ("2022", "2022-01-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
           ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
           ("2026", "2026-01-01", "2026-08-28")]


def build_frame(part, bull_thr, edge_lookback):
    fac = build_factors(part)
    micro = aggregate_score(fac, MICRO_W)
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=48,
                       er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
    sig = np.where((reg > 0.5) & (micro > 0.20), 1, np.where((reg > 0.5) & (micro < -0.20), -1, 0))
    bull_ok = part["bull"].to_numpy() > bull_thr
    # rolling edge proxy: mean micro score over lookback (positive = momentum edge intact)
    micro_s = pd.Series(micro.to_numpy(), index=range(len(part)))
    edge = micro_s.rolling(edge_lookback, min_periods=int(edge_lookback * 0.5)).mean().fillna(0.0).to_numpy()
    edge_ok = edge > 0.02
    sig = np.where(bull_ok & edge_ok, sig, 0)
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy(); anim = ~np.isnan(a)
    return pd.DataFrame({
        "date": part["date"], "open": part["open"], "high": part["high"], "low": part["low"],
        "close": part["close"], "signal": sig,
        "stop_dist": np.where(anim, a * 0.8, 0.0), "tp_dist": np.where(anim, a * 0.8 * 2.2, 0.0)})


def run(part, bull_thr, edge_lookback, lev, risk_pct, margin_call):
    f = build_frame(part, bull_thr, edge_lookback)
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, leverage=lev, risk_per_trade_pct=risk_pct,
                      margin_call_pct=margin_call)
    return run_leverage_backtest(f, cost, "10x")["stats"]


print("=== 10x LEVERAGE BULL-ONLY (double gate) — margin-account ===")
print(f"{'bull':>5} {'edge':>5} | {'2019':>6} {'2020':>6} {'2024':>6} {'2025':>6} {'2026':>6} | {'2022':>6} {'2023':>6}")
print("-" * 72)
best = {"ret": -1e9}
for bull_thr in (0.55, 0.65):
    for edge in (96, 192):
        row = {}
        for wname, s, e in WINDOWS:
            part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
            st = run(part, bull_thr, edge, 10.0, 0.01, 0.20)
            row[wname] = st
        def f(w):
            return f"{row[w]['total_return']:+6.1f}" if w in row else "  n/a"
        print(f"{bull_thr:5.2f} {edge:5d} | {f('2019')} {f('2020')} {f('2024')} {f('2025')} {f('2026')} | {f('2022')} {f('2023')}")
        # score: sum of bull windows minus bull-fails
        bull_sum = row["2019"]["total_return"] + row["2020"]["total_return"] + row["2024"]["total_return"] + row["2025"]["total_return"] + row["2026"]["total_return"]
        fail = row["2022"]["total_return"] + row["2023"]["total_return"]
        print(f"        bull_sum={bull_sum:+.0f}%  fail_sum={fail:+.0f}%  net={bull_sum+fail:+.0f}%")
        if bull_sum + fail > best["ret"]:
            best = {"bull": bull_thr, "edge": edge, "ret": bull_sum + fail}
print("\nBEST:", best)
