# -*- coding: utf-8 -*-
"""Final recommended high-leverage bull strategy: 3x leverage + bull gate.
Year-by-year, and full bull era report, saved equity/trades."""
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


def run(part, lev, thr=0.5, stop=2.5, rr=2.0, mc=0.30, bull_thr=0.55):
    fac = build_factors(part)
    micro = aggregate_score(fac, MICRO_W)
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=48,
                       er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
    sig = np.where((reg > 0.5) & (micro > thr), 1, np.where((reg > 0.5) & (micro < -thr), -1, 0))
    sig = np.where(part["bull"].to_numpy() > bull_thr, sig, 0)
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy(); anim = ~np.isnan(a)
    f = pd.DataFrame({"date": part["date"], "open": part["open"], "high": part["high"],
                      "low": part["low"], "close": part["close"], "signal": sig,
                      "stop_dist": np.where(anim, a * stop, 0.0),
                      "tp_dist": np.where(anim, a * stop * rr, 0.0)})
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, leverage=lev, risk_per_trade_pct=0.20,
                      margin_call_pct=mc)
    return run_leverage_backtest(f, cost, "hbm")["stats"]


print("=== RECOMMENDED 3x LEVERAGE BULL STRATEGY (bull gate, long+short) ===")
print(f"{'year':>5} {'bull%':>5} | {'ret%':>8} {'PF':>5} {'maxDD%':>6}")
print("-" * 40)
for wname, s, e in WINDOWS:
    part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
    st = run(part, 3.0)
    bull_pct = part["bull"].mean() * 100
    print(f"{wname:>5} {bull_pct:5.1f} | {st['total_return']:+8.1f} {st['profit_factor']:5.2f} {st['max_drawdown']:6.1f}")

print()
print("=== FULL BULL ERA 2024-2026 @ 3x ===")
era = DF[(DF["date"] >= "2024-01-01") & (DF["date"] <= "2026-08-28")].reset_index(drop=True)
st = run(era, 3.0)
print(f"ret={st['total_return']:+.1f}%  PF={st['profit_factor']:.2f}  maxDD={st['max_drawdown']:.1f}%  final=${st['final_equity']:,.0f}")
