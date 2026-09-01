# -*- coding: utf-8 -*-
"""Replay micro vs three-line under a bull-regime gate (long-only in bull).

Principle: only trade in BULL regime; stand aside (flat) in CHOP/BEAR.
Compares micro-only, three-line, and their bull-gated variants by window.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import adx, atr, ema, trend_regime
from london_gold.gold_system import SystemConfig, build_three_line_frame

spec = importlib.util.spec_from_file_location(
    "v3bt", str(Path(__file__).resolve().parents[1] / "scripts" / "london_gold_intraday_v3_backtest.py"))
v3bt = importlib.util.module_from_spec(spec)
sys.modules["v3bt"] = v3bt
spec.loader.exec_module(v3bt)

DF = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\XAUUSD_1h_continuous.csv")
DF["date"] = pd.to_datetime(DF["date"], utc=True)
DF = DF.sort_values("date").reset_index(drop=True)
MACRO = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\macro_daily.csv", parse_dates=["date"])
COT = pd.read_csv(r"C:\Users\张策\Documents\EA量化项目\data\gold_cot_disagg.csv", parse_dates=["date"])
COST = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                  commission_per_oz=0.10, risk_per_trade_pct=0.01)

# Bull score (from explore_bull_bear) as a column on DF
close = DF["close"]
ema200 = ema(close, 4800)
ema_slope = ema200.diff(120)
up_trend = (close > ema(close, 1200)).astype(float)
trend = trend_regime(close, DF["high"], DF["low"], er_window=48, er_threshold=0.12,
                     adx_window=14, adx_threshold=20.0)
DF["bull"] = np.clip((ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3, 0, 1).to_numpy()


def gate_signals(frame, bull_filter):
    """Zero out signals where bull gate is off."""
    f = frame.copy()
    f["signal"] = np.where(bull_filter, f["signal"], 0)
    return f


def run_micro(part, use_bull_gate, gate_filter):
    fac = build_factors(part)
    micro = aggregate_score(fac, {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
                                  "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4})
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=48,
                       er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
    sig = np.where((reg > 0.5) & (micro > 0.25), 1, np.where((reg > 0.5) & (micro < -0.25), -1, 0))
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy()
    anim = ~np.isnan(a)
    f = pd.DataFrame({
        "date": part["date"], "open": part["open"], "high": part["high"],
        "low": part["low"], "close": part["close"],
        "signal": sig, "stop_dist": np.where(anim, a * 0.8, 0.0), "tp_dist": np.where(anim, a * 0.8 * 2.2, 0.0),
    })
    if use_bull_gate:
        f = gate_signals(f, gate_filter)
    return v3bt.backtest_v3(f, COST, "micro", {}).stats


WINDOWS = [
    ("2018", "2018-01-01", "2018-12-31"), ("2019", "2019-01-01", "2019-12-31"),
    ("2020", "2020-01-01", "2020-12-31"), ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-08-28"),
]

print(f"{'year':>5} | {'micro_ret':>9} {'mDD':>5} | {'bullGated_ret':>13} {'bDD':>5} | {'bull%':>5} label")
print("-" * 64)
for wname, s, e in WINDOWS:
    part = DF[(DF["date"] >= s) & (DF["date"] <= e)].reset_index(drop=True)
    gate = part["bull"].to_numpy() > 0.55
    m = run_micro(part, use_bull_gate=False, gate_filter=None)
    mg = run_micro(part, use_bull_gate=True, gate_filter=gate) if len(part) else m
    bull_pct = gate.mean() * 100 if len(gate) else 0
    lab = "BULL" if bull_pct > 55 else ("CHOP" if bull_pct > 35 else "BEAR")
    print(f"{wname:>5} | {m['total_return']:+9.1f} {m['max_drawdown']:5.1f} | "
          f"{mg['total_return']:+13.1f} {mg['max_drawdown']:5.1f} | {bull_pct:5.1f} {lab}")
