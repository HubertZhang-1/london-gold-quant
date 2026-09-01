# -*- coding: utf-8 -*-
"""Maximize bull-market returns: micro strategy tuned for BULL regimes only.

Only trade when bull_score > threshold (bull regime), stand aside otherwise.
In bull regimes, test higher leverage / looser entry / larger sizing to
amplify returns. Replay key windows.
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

spec = importlib.util.spec_from_file_location(
    "v3bt", str(Path(__file__).resolve().parents[1] / "scripts" / "london_gold_intraday_v3_backtest.py"))
v3bt = importlib.util.module_from_spec(spec)
sys.modules["v3bt"] = v3bt
spec.loader.exec_module(v3bt)

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

WINDOWS = [
    ("2018", "2018-01-01", "2018-12-31"), ("2019", "2019-01-01", "2019-12-31"),
    ("2020", "2020-01-01", "2020-12-31"), ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-08-28"),
]


def run(part, bull_gate, thr, stop, rr, lev):
    fac = build_factors(part)
    micro = aggregate_score(fac, {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
                                  "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4})
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=48,
                       er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
    sig = np.where((reg > 0.5) & (micro > thr), 1, np.where((reg > 0.5) & (micro < -thr), -1, 0))
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy()
    anim = ~np.isnan(a)
    f = pd.DataFrame({
        "date": part["date"], "open": part["open"], "high": part["high"],
        "low": part["low"], "close": part["close"],
        "signal": sig, "stop_dist": np.where(anim, a * stop, 0.0),
        "tp_dist": np.where(anim, a * stop * rr, 0.0)})
    if bull_gate:
        f["signal"] = np.where(part["bull"].to_numpy() > 0.55, f["signal"], 0)
    cost = CostConfig(capital=100_000, position_oz=10 * lev, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, risk_per_trade_pct=0.01)
    return v3bt.backtest_v3(f, cost, "b", {}).stats


print("=== Bull-gated micro: base vs amplified (higher leverage) ===")
print(f"{'year':>5} | {'base_ret':>8} {'baseDD':>6} | {'lev2_ret':>9} {'lev2DD':>6} | {'lev3_ret':>9} {'lev3DD':>6}")
print("-" * 70)
for wname, s, e in WINDOWS:
    part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
    base = run(part, True, 0.25, 0.8, 2.2, 1.0)
    lev2 = run(part, True, 0.20, 0.8, 2.2, 2.0)
    lev3 = run(part, True, 0.15, 0.8, 2.2, 3.0)
    print(f"{wname:>5} | {base['total_return']:+8.1f} {base['max_drawdown']:6.1f} | "
          f"{lev2['total_return']:+9.1f} {lev2['max_drawdown']:6.1f} | "
          f"{lev3['total_return']:+9.1f} {lev3['max_drawdown']:6.1f}")
