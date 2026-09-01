# -*- coding: utf-8 -*-
"""Comprehensive bull-era replay: micro / three-line / v3 under bull gate."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from london_gold.backtest import CostConfig
from london_gold.factor_library import aggregate_score, build_factors
from london_gold.indicators import atr, ema, trend_regime
from london_gold.gold_system import SystemConfig, build_three_line_frame
from london_gold.intraday_strategies_v3 import momentum_scalp_signals

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
close = DF["close"]
ema200 = ema(close, 4800)
ema_slope = ema200.diff(120)
up_trend = (close > ema(close, 1200)).astype(float)
trend = trend_regime(close, DF["high"], DF["low"], er_window=48, er_threshold=0.12,
                     adx_window=14, adx_threshold=20.0)
DF["bull"] = np.clip((ema_slope > 0).astype(float) * 0.4 + up_trend * 0.3 + trend * 0.3, 0, 1)

# Continuous BULL-era (2024-2026) - the target window for a bull strategy
BULL_ERA = DF[(DF["date"] >= "2024-01-01") & (DF["date"] <= "2026-08-28")].reset_index(drop=True)


def run_micro(part, lev):
    fac = build_factors(part)
    micro = aggregate_score(fac, {"macd": 1.3, "aroon": 1.3, "trend_adx": 1.2, "ema_spread": 1.0,
                                  "bulls_bears": 0.9, "momentum": 1.0, "bb_position": 0.4})
    reg = trend_regime(part["close"], part["high"], part["low"], er_window=48,
                       er_threshold=0.12, adx_window=14, adx_threshold=20.0).to_numpy()
    sig = np.where((reg > 0.5) & (micro > 0.20), 1, np.where((reg > 0.5) & (micro < -0.20), -1, 0))
    sig = np.where(part["bull"].to_numpy() > 0.55, sig, 0)
    a = atr(part["high"], part["low"], part["close"], 14).to_numpy(); anim = ~np.isnan(a)
    f = pd.DataFrame({"date": part["date"], "open": part["open"], "high": part["high"],
                      "low": part["low"], "close": part["close"], "signal": sig,
                      "stop_dist": np.where(anim, a * 0.8, 0.0), "tp_dist": np.where(anim, a * 0.8 * 2.2, 0.0)})
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, risk_per_trade_pct=0.01 * lev)
    return v3bt.backtest_v3(f, cost, "micro", {}).stats


def run_three(part, lev):
    cfg = SystemConfig(cot_weight=1.0, macro_weight=0.3, micro_threshold=0.15)
    frame = build_three_line_frame(part, COT, MACRO, cfg)
    frame["signal"] = np.where(part["bull"].to_numpy() > 0.55, frame["signal"], 0)
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, risk_per_trade_pct=0.01 * lev)
    return v3bt.backtest_v3(frame, cost, "three", {}).stats


def run_v3(part, lev):
    frame = momentum_scalp_signals(part, min_confidence=0.4, rr_target=2.5, stop_mult=0.8)
    frame["signal"] = np.where(part["bull"].to_numpy() > 0.55, frame["signal"], 0)
    cost = CostConfig(capital=100_000, position_oz=10, spread=0.35, slippage=0.10,
                      commission_per_oz=0.10, risk_per_trade_pct=0.01 * lev)
    return v3bt.backtest_v3(frame, cost, "v3", {}).stats


print("=== BULL-ERA 2024-2026 (bull-gated) strategies x leverage ===")
print(f"{'strategy':>16} {'lev':>4} | {'trades':>6} {'ret%':>8} {'PF':>5} {'maxDD%':>6}")
print("-" * 58)
for name, fn in [("micro", run_micro), ("three-line", run_three), ("v3_momentum", run_v3)]:
    for lev in (1.0, 2.0):
        s = fn(BULL_ERA, lev)
        print(f"{name:>16} {lev:4.1f} | {s['trade_count']:6d} {s['total_return']:+8.1f} "
              f"{s['profit_factor']:5.2f} {s['max_drawdown']:6.1f}")
